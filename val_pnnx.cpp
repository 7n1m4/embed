// Stock-ncnn CPU runner for the pnnx-generated microsoft/unixcoder-base graph.
//
// NO custom layers: the pnnx graph is 100% stock ncnn ops (Embed/LayerNorm/
// InnerProduct/SDPA/GELU/Reshape/Permute/Split/BinaryOp) because every
// data-dependent tensor is fed in as a model input.
//
// Inputs (ncnn Mat w=seq h=batch c=1, row(s)[b] layout):
//   in0 = input_ids     int32 [batch, seq]
//   in1 = attn_mask_4d  f32   [batch, 1, 1, seq]   (0.0 attend, MASK_FILL masked)
//   in2 = position_ids  int32 [batch, seq]
//   in3 = token_type_ids int32 [batch, seq]  (zeros)
// Output:
//   out0 = last_hidden_state f32 [batch, seq, 768] -> mat w=768 h=seq c=batch
//
// Mean-pools over the non-pad positions and compares against the torch
// reference in unixcoder-onnx/reference_embeddings.json.
//
// Build:
//   g++ -std=c++11 -O2 -o val_pnnx val_pnnx.cpp -I/usr/include/ncnn -lncnn
// Run:
//   ./val_pnnx
//   ./val_pnnx unixcoder-onnx/unixcoder.pnnx.fp32.ncnn.param unixcoder-onnx/unixcoder.pnnx.fp32.ncnn.bin
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <fstream>
#include <sstream>
#include <string>
#include <vector>

#include "net.h"
#include "mat.h"

static const char* kParam = "unixcoder-onnx/unixcoder.pnnx.fp32.ncnn.param";
static const char* kBin = "unixcoder-onnx/unixcoder.pnnx.fp32.ncnn.bin";
static const char* kDir = "unixcoder-onnx/";
static const char* kRef = "unixcoder-onnx/reference_embeddings.json";

// ---- minimal .npy reader (little-endian, no pickle) -----------------------
static bool read_npy(const std::string& path, std::vector<int>& shape,
                     std::string& descr, std::vector<unsigned char>& out)
{
    std::ifstream f(path, std::ios::binary);
    if (!f) return false;
    char magic[6];
    f.read(magic, 6);
    if (std::memcmp(magic, "\x93NUMPY", 6) != 0) return false;
    uint8_t ver[2];
    f.read((char*)ver, 2);
    uint32_t hlen = 0;
    if (ver[0] == 1)
    {
        uint16_t l;
        f.read((char*)&l, 2);
        hlen = l;
    }
    else
    {
        uint32_t l;
        f.read((char*)&l, 4);
        hlen = l;
    }
    std::string header;
    header.resize(hlen);
    f.read(&header[0], hlen);

    size_t q1 = header.find('\'', header.find("'descr'") + 7);
    size_t q2 = header.find('\'', q1 + 1);
    descr = (q1 != std::string::npos && q2 != std::string::npos)
                ? header.substr(q1 + 1, q2 - q1 - 1) : "<f4";

    shape.clear();
    size_t sp = header.find("'shape'");
    if (sp != std::string::npos)
    {
        size_t lp = header.find('(', sp);
        size_t rp = header.find(')', lp);
        std::stringstream ss(header.substr(lp + 1, rp - lp - 1));
        std::string tok;
        while (std::getline(ss, tok, ','))
        {
            if (tok.empty()) continue;
            shape.push_back(atoi(tok.c_str()));
        }
    }
    if (shape.empty()) shape.push_back(0);
    out.assign((std::istreambuf_iterator<char>(f)), std::istreambuf_iterator<char>());
    // bound to the declared element count so we never read trailing junk
    {
        int64_t n = 1;
        for (size_t k = 0; k < shape.size(); k++) n *= (int64_t)shape[k];
        size_t itemsize = (descr.size() >= 3 && descr[2] == '8') ? 8 : 4;
        size_t want = (size_t)(n * itemsize);
        if (out.size() > want) out.resize(want);
        if (out.size() < want)
        {
            out.resize(want, 0); // pad; caller validates via shapes
        }
    }
    return true;
}

static inline int32_t npz_int(const std::vector<unsigned char>& d, size_t i, const std::string& descr)
{
    if (descr == "<i8") return (int32_t)((const int64_t*)d.data())[i];
    if (descr == "<i4") return ((const int32_t*)d.data())[i];
    if (descr == "<i2") return (int32_t)((const int16_t*)d.data())[i];
    if (descr == "<i1") return (int32_t)((const int8_t*)d.data())[i];
    return ((const int32_t*)d.data())[i];
}

static bool read_ref_json(const std::string& path, std::vector<std::vector<float> >& refs)
{
    std::ifstream f(path);
    if (!f) return false;
    std::stringstream ss;
    ss << f.rdbuf();
    std::string s = ss.str();
    size_t k = s.find("\"embeddings\"");
    if (k == std::string::npos) return false;
    size_t lb = s.find('[', k);
    if (lb == std::string::npos) return false;
    refs.clear();
    size_t i = lb;
    int depth = 0;
    while (i < s.size())
    {
        char c = s[i];
        if (c == '[') { depth++; if (depth == 1 && !refs.empty()) refs.push_back(std::vector<float>()); i++; continue; }
        if (c == ']') { depth--; if (depth == 0) break; i++; continue; }
        if (c == ',' || c == ' ' || c == '\n' || c == '\t' || c == '\r') { i++; continue; }
        if (std::isdigit((unsigned char)c) || c == '-' || c == '+' || c == '.')
        {
            size_t start = i;
            while (i < s.size() && (std::isdigit((unsigned char)s[i]) || strchr("-+.eE", s[i]) && s[i] != '\0')) i++;
            float v = (float)atof(s.substr(start, i - start).c_str());
            if (depth >= 3) refs.back().push_back(v);
            else if (depth == 2) { if (refs.empty()) refs.push_back(std::vector<float>()); refs.back().push_back(v); }
            continue;
        }
        i++;
    }
    return !refs.empty();
}

int main(int argc, char** argv)
{
    const char* param_path = argc > 1 ? argv[1] : kParam;
    const char* bin_path = argc > 2 ? argv[2] : kBin;
    const char* ref_path = argc > 4 ? argv[4] : kRef;

    std::vector<int> sh_ids, sh_pos, sh_tts, sh_m2d, sh_m4;
    std::vector<unsigned char> raw_ids, raw_pos, raw_tts, raw_m2d, raw_am4;
    std::string descr_ids, descr_pos, descr_tts, descr_m4;

    const std::string kdir = kDir;
    if (!read_npy(kdir + "fixture_input_ids.npy", sh_ids, descr_ids, raw_ids))
    { fprintf(stderr, "cannot read fixture_input_ids.npy\n"); return 1; }
    const int batch = sh_ids[0], seq = sh_ids[1];
    read_npy(kdir + "fixture_position_ids.npy", sh_pos, descr_pos, raw_pos);
    read_npy(kdir + "fixture_token_type_ids.npy", sh_tts, descr_tts, raw_tts);
    read_npy(kdir + "fixture_attention_mask.npy", sh_m2d, descr_m4, raw_m2d);
    if (!read_npy(kdir + "fixture_attn_mask_4d.npy", sh_m4, descr_m4, raw_am4))
    { fprintf(stderr, "cannot read fixture_attn_mask_4d.npy\n"); return 1; }

    printf("fixture: batch=%d seq=%d  attn_mask_4d=[%d,%d,%d,%d]\n", batch, seq,
           sh_m4[0], sh_m4[1], sh_m4[2], sh_m4[3]);

    std::vector<std::vector<float> > refs;
    if (!read_ref_json(ref_path, refs)) { fprintf(stderr, "bad reference json\n"); return 1; }
    const int D = 768;

    ncnn::Net net;
    net.opt.use_vulkan_compute = false;
    net.opt.use_fp16_packed = false;
    net.opt.use_fp16_storage = false;
    net.opt.use_fp16_arithmetic = false;
    net.opt.use_packing_layout = false;
    net.opt.num_threads = 1;
    net.opt.openmp_blocktime = 20;
    if (net.load_param(param_path) != 0) { fprintf(stderr, "load_param failed\n"); return 1; }
    if (net.load_model(bin_path) != 0) { fprintf(stderr, "load_model failed\n"); return 1; }
    printf("net loaded: param=%s\nlayers=%zu\n", param_path, net.layers().size());

    // helper: fill an existing [seq, batch, 4u] mat in place
    // (ncnn::Mat must NOT be returned by value in this ncnn build: the
    //  fastmemcpy/refcount path corrupts the heap on the next allocation)
    auto fill_ids = [&](ncnn::Mat& m, const std::vector<unsigned char>& d) {
        for (int b = 0; b < batch; b++)
        {
            ncnn::Mat ch = m.channel(b);
            for (int s = 0; s < seq; s++)
                ch.row(s)[0] = (float)npz_int(d, (size_t)b * seq + s, descr_ids);
        }
    };

    ncnn::Mat in0(seq, 1, batch, (size_t)4u, 1);
    fill_ids(in0, raw_ids);
    ncnn::Mat in2(seq, 1, batch, (size_t)4u, 1);
    fill_ids(in2, raw_pos);
    ncnn::Mat in3(seq, 1, batch, (size_t)4u, 1);
    fill_ids(in3, raw_tts);

    // 4D mask [batch,1,1,seq] -> ncnn mat w=seq h=batch c=1 (channel = batch)
    ncnn::Mat in1(seq, 1, batch, (size_t)4u, 1);
    {
        const float* src = (const float*)raw_am4.data();
        // npz C-order: [batch][1][1][seq] -> offset = b*seq + s
        for (int b = 0; b < batch; b++)
        {
            ncnn::Mat ch = in1.channel(b);
            for (int s = 0; s < seq; s++)
                ch.row(s)[0] = src[(size_t)b * seq + s];
        }
    }

    ncnn::Extractor ex = net.create_extractor();
    ex.input("in0", in0);
    ex.input("in1", in1);
    ex.input("in2", in2);
    ex.input("in3", in3);

    ncnn::Mat out;
    if (ex.extract("out0", out) != 0) { fprintf(stderr, "extract out0 failed\n"); return 1; }
    printf("output: w=%d h=%d c=%d (expect w=768 h=seq c=batch)\n\n", out.w, out.h, out.c);
    if (out.empty()) { fprintf(stderr, "empty output\n"); return 1; }

    // ncnn SDPA output is [seq][768][batch] -> channel(b).row(s)
    printf("%-4s %-14s %-14s %-14s\n", "row", "max_abs_diff", "mean_abs", "cosine");
    double worst_diff = 0.0, worst_cos = 1.0;
    bool flat = (refs.size() != (size_t)batch);
    for (int b = 0; b < batch; b++)
    {
        std::vector<double> pooled(D, 0.0);
        int ntok = 0;
        for (int s = 0; s < seq && s < out.h; s++)
        {
            if (in1.row(s)[b] != 0.f) continue;
            const float* p = out.channel(b).row(s);
            for (int d = 0; d < D && d < out.w; d++) pooled[d] += p[d];
            ntok++;
        }
        if (ntok == 0) { printf("%-4d %-14s %-14s %-14s\n", b, "EMPTY", "-", "-"); continue; }
        for (int d = 0; d < D; d++) pooled[d] /= (double)ntok;

        const float* r = flat ? &refs[0][(size_t)b * D] : &refs[b][0];
        double maxdiff = 0.0, mean = 0.0, dot = 0.0, na = 0.0, nb = 0.0;
        for (int d = 0; d < D; d++)
        {
            double dd = std::fabs(pooled[d] - (double)r[d]);
            if (dd > maxdiff) maxdiff = dd;
            mean += dd;
            dot += pooled[d] * (double)r[d];
            na += pooled[d] * pooled[d];
            nb += (double)r[d] * (double)r[d];
        }
        double cos = dot / (std::sqrt(na) * std::sqrt(nb));
        printf("%-4d %-14.3e %-14.3e %-14.6f\n", b, maxdiff, mean / D, cos);
        if (maxdiff > worst_diff) worst_diff = maxdiff;
        if (cos < worst_cos) worst_cos = cos;
    }
    printf("\nWORST: max_abs_diff=%.3e  min_cosine=%.6f\n", worst_diff, worst_cos);
    return 0;
}
