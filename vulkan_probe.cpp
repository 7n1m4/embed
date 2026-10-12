// vulkan_probe.cpp
//
// Minimal ncnn + Vulkan probe.
//
// Goal: prove that a workload can be *explicitly* pinned to the second discrete
// NVIDIA GPU (the GTX 1050, 2GB) instead of defaulting to device 0 (the GTX 1060,
// which is driven by a live gaming workload and must never be touched).
//
// Device selection is done by *name*, never by hardcoded index: both cards are
// discrete NVIDIA (Vulkan device 0 = GTX 1060, device 1 = GTX 1050), so an
// implicit default would land on the 1060. The probe searches for the device
// whose VkPhysicalDeviceProperties.deviceName contains "GTX 1050" and only ever
// constructs a VulkanDevice for that index.
//
// The actual dispatch is a real compute pipeline (the ReLU shader from ncnn's
// own shader tree, compiled through ncnn's Pipeline/PipelineCache machinery),
// run on data uploaded into device memory and read back, so a successful run
// proves both device attachment and real GPU execution.
//
// Exit code 0 only if the pinned device is used and the compute result matches
// the CPU reference.

#include "command.h"
#include "cpu.h"
#include "gpu.h"
#include "layer.h"
#include "layer_shader_type.h"
#include "mat.h"
#include "option.h"
#include "pipeline.h"

#include <stdio.h>
#include <string.h>
#include <string>
#include <vector>

// The exact device we are allowed to use. Selected by substring so a driver or
// PCI enumeration change cannot silently fall through to the other card.
static const char* PINNED_DEVICE_NAME = "GTX 1050";

int main()
{
    // ---------------------------------------------------------------
    // 1. Enumerate every Vulkan device ncnn can see.
    // ---------------------------------------------------------------
    // create_gpu_instance() creates the VkInstance, walks all VkPhysicalDevices
    // and populates the internal GpuInfo table. It does NOT create a VkDevice on
    // any card, so enumeration alone is harmless.
    ncnn::create_gpu_instance();

    const int gpu_count = ncnn::get_gpu_count();
    printf("ncnn::get_gpu_count() = %d\n", gpu_count);
    if (gpu_count <= 0)
    {
        fprintf(stderr, "FATAL: no vulkan devices visible to ncnn\n");
        return 1;
    }

    int pinned_index = -1;
    printf("---- enumerated vulkan devices ----\n");
    for (int i = 0; i < gpu_count; i++)
    {
        const ncnn::GpuInfo& info = ncnn::get_gpu_info(i);

        const VkPhysicalDeviceProperties& props = info.physicalDeviceProperties();
        const char* name = info.device_name();

        // device type 0/discrete, 1/integrated, 2/virtual, 3/cpu
        printf("  device %d: type=%d vendor_id=0x%04x device_id=0x%04x api_version=%u.%u.%u driver_id=%u driver_name=\"%s\" name=\"%s\"\n",
                i, info.type(), info.vendor_id(), info.device_id(),
                VK_VERSION_MAJOR(info.api_version()), VK_VERSION_MINOR(info.api_version()), VK_VERSION_PATCH(info.api_version()),
                info.driver_id(), info.driver_name(), name);

        // remember the first device whose name matches the pinned card
        if (pinned_index < 0 && strstr(name, PINNED_DEVICE_NAME))
        {
            pinned_index = i;
        }
    }
    printf("---------------------------------\n");

    if (pinned_index < 0)
    {
        fprintf(stderr, "FATAL: device \"%s\" not found among the %d vulkan device(s)\n", PINNED_DEVICE_NAME, gpu_count);
        ncnn::destroy_gpu_instance();
        return 1;
    }

    printf("selected vulkan device index %d (name substring \"%s\")\n", pinned_index, PINNED_DEVICE_NAME);
    if (pinned_index == 0)
    {
        // No ambiguity is tolerated here: device 0 is the 1060 in this machine
        // layout, and attaching compute work to it is forbidden.
        fprintf(stderr, "FATAL: refusing to run on vulkan device 0, expected a non-zero index for the pinned device\n");
        ncnn::destroy_gpu_instance();
        return 1;
    }

    const ncnn::GpuInfo& pinned_info = ncnn::get_gpu_info(pinned_index);
    printf("pinned device: name=\"%s\" type=%d vendor=0x%04x device=0x%04x api=%u.%u.%u driver=%s\n",
            pinned_info.device_name(), pinned_info.type(),
            pinned_info.vendor_id(), pinned_info.device_id(),
            VK_VERSION_MAJOR(pinned_info.api_version()), VK_VERSION_MINOR(pinned_info.api_version()), VK_VERSION_PATCH(pinned_info.api_version()),
            pinned_info.driver_name());
    printf("pinned device heaps: %d memory heap(s); max_workgroup_invocations=%u subgroup_size=%u\n",
            (int)pinned_info.physicalDeviceMemoryProperties().memoryHeapCount,
            pinned_info.max_workgroup_invocations(), pinned_info.subgroup_size());

    // ---------------------------------------------------------------
    // 2. Pin: obtain the VulkanDevice for that specific index.
    // ---------------------------------------------------------------
    // This is the whole trick. get_gpu_device(index) lazily constructs the
    // VulkanDevice (a full VkDevice, queues, allocators, pipeline cache) for
    // that one physical device and caches it. Nothing else is instantiated, so
    // the 1060 never gets a VkDevice and never sees any work.
    //
    // Note: get_default_gpu_index() is deliberately NOT used.
    ncnn::VulkanDevice* vkdev = ncnn::get_gpu_device(pinned_index);
    if (!vkdev || !vkdev->is_valid())
    {
        fprintf(stderr, "FATAL: failed to create vulkan device for index %d\n", pinned_index);
        ncnn::destroy_gpu_instance();
        return 1;
    }
    printf("obtained ncnn::VulkanDevice for index %d, VkDevice=%p\n", pinned_index, (void*)vkdev->vkdevice());
    if (vkdev->info.device_index() != pinned_index)
    {
        fprintf(stderr, "FATAL: vulkan device index mismatch: asked %d, got %d\n",
                pinned_index, vkdev->info.device_index());
        ncnn::destroy_gpu_instance();
        return 1;
    }
    if (strstr(vkdev->info.device_name(), PINNED_DEVICE_NAME) == 0)
    {
        fprintf(stderr, "FATAL: vulkan device name mismatch: asked \"%s\", got \"%s\"\n",
                PINNED_DEVICE_NAME, vkdev->info.device_name());
        ncnn::destroy_gpu_instance();
        return 1;
    }

    // Option carries the device index too, so any layer/allocator/pipeline
    // created with it resolves to the same card (not the default one).
    ncnn::Option opt;
    opt.use_vulkan_compute = true;
    opt.use_fp16_packed = false;
    opt.use_fp16_storage = false;
    opt.use_fp16_arithmetic = false;
    opt.use_int8_packed = false;
    opt.use_int8_storage = false;
    opt.use_bf16_storage = false;
    opt.vulkan_device_index = pinned_index;
    printf("configured Option: use_vulkan_compute=true vulkan_device_index=%d\n", opt.vulkan_device_index);

    // Allocators are explicitly bound to the pinned device (never the default).
    ncnn::VkAllocator* blob_vkallocator = vkdev->acquire_blob_allocator();
    ncnn::VkAllocator* staging_vkallocator = vkdev->acquire_staging_allocator();
    opt.blob_vkallocator = blob_vkallocator;
    opt.workspace_vkallocator = blob_vkallocator;
    opt.staging_vkallocator = staging_vkallocator;

    // ReLU layer whose vulkan backend lives on the pinned device.
    ncnn::Layer* op = ncnn::create_layer_vulkan("ReLU");
    if (!op)
    {
        fprintf(stderr, "FATAL: ReLU has no vulkan implementation\n");
        vkdev->reclaim_staging_allocator(staging_vkallocator);
        vkdev->reclaim_blob_allocator(blob_vkallocator);
        ncnn::destroy_gpu_instance();
        return 1;
    }
    ncnn::ParamDict pd;
    pd.set(0, 0.f); // slope = 0  =>  v = max(v, 0)
    op->load_param(pd);
    op->vkdev = vkdev;

    // Build the compute pipeline; this compiles the relu.comp spir-v for this
    // specific device.
    if (op->create_pipeline(opt) != 0)
    {
        fprintf(stderr, "FATAL: ReLU create_pipeline failed\n");
        delete op;
        vkdev->reclaim_staging_allocator(staging_vkallocator);
        vkdev->reclaim_blob_allocator(blob_vkallocator);
        ncnn::destroy_gpu_instance();
        return 1;
    }

    // ---------------------------------------------------------------
    // 3. Real device round-trip: host Mat -> VkMat -> shader -> VkMat -> Mat.
    // ---------------------------------------------------------------
    // 1024 floats, packed 4-wide so the shader's vec4 path is exercised.
    // NOTE for packed Mats, elemsize = 4 * elempack (bytes of one packed element),
    // and w is the packed (divided-by-4) width. cstep = alignSize(w*elemsize,16)/elemsize,
    // so total() = cstep covers exactly elempack*w = 1024 floats contiguously.
    const int elems = 1024;
    const int elempack = 4;
    const size_t elemsize_packed = (size_t)(4u * elempack); // bytes per packed element
    ncnn::Mat bottom(elems / elempack, elemsize_packed, elempack);
    if (bottom.empty())
    {
        fprintf(stderr, "FATAL: could not allocate host Mat\n");
        op->destroy_pipeline(opt);
        delete op;
        vkdev->reclaim_staging_allocator(staging_vkallocator);
        vkdev->reclaim_blob_allocator(blob_vkallocator);
        ncnn::destroy_gpu_instance();
        return 1;
    }
    if (bottom.total() != (size_t)(elems / elempack) || bottom.elemsize != elemsize_packed)
    {
        fprintf(stderr, "FATAL: host Mat layout unexpected: total=%zu elemsize=%zu\n", bottom.total(), bottom.elemsize);
        op->destroy_pipeline(opt);
        delete op;
        vkdev->reclaim_staging_allocator(staging_vkallocator);
        vkdev->reclaim_blob_allocator(blob_vkallocator);
        ncnn::destroy_gpu_instance();
        return 1;
    }

    // Known pattern: alternating sign, so max(v,0) zeros exactly half.
    float* p = bottom;
    float expected_sum = 0.f;
    for (int i = 0; i < elems; i++)
    {
        p[i] = ((i % 2 == 0) ? 1.f : -1.f) * float(i);
        expected_sum += (i % 2 == 0) ? float(i) : 0.f;
    }
    printf("host input: %d floats, expected relu output sum = %.1f\n", elems, expected_sum);

    {
        // VkCompute records on the pinned device's compute queue.
        ncnn::VkCompute cmd(vkdev);

        ncnn::VkMat top;
        ncnn::Mat downloaded;

        // upload host -> device local
        cmd.record_upload(bottom, top, opt);

        // run the shader in-place on device memory
        if (op->forward_inplace(top, cmd, opt) != 0)
        {
            fprintf(stderr, "FATAL: vulkan forward_inplace failed\n");
            delete op;
            vkdev->reclaim_staging_allocator(staging_vkallocator);
            vkdev->reclaim_blob_allocator(blob_vkallocator);
            ncnn::destroy_gpu_instance();
            return 1;
        }

        // download device -> host
        cmd.record_download(top, downloaded, opt);

        // dispatch + wait for the GPU
        const int ret = cmd.submit_and_wait();
        if (ret != 0)
        {
            fprintf(stderr, "FATAL: submit_and_wait returned %d\n", ret);
            delete op;
            vkdev->reclaim_staging_allocator(staging_vkallocator);
            vkdev->reclaim_blob_allocator(blob_vkallocator);
            ncnn::destroy_gpu_instance();
            return 1;
        }

        // ncnn's download keeps packing (dims=1, w=elems/elempack, elemsize=4*elempack),
        // so unpack to plain fp32 before comparing.
        if (downloaded.empty() || downloaded.dims != 1
                || downloaded.elempack != elempack
                || downloaded.w * downloaded.elempack != elems)
        {
            fprintf(stderr, "FATAL: download produced empty/wrong result (dims=%d w=%d elempack=%d elemsize=%zu)\n",
                    downloaded.dims, downloaded.w, downloaded.elempack, downloaded.elemsize);
            delete op;
            vkdev->reclaim_staging_allocator(staging_vkallocator);
            vkdev->reclaim_blob_allocator(blob_vkallocator);
            ncnn::destroy_gpu_instance();
            return 1;
        }

        ncnn::Mat result;
        ncnn::convert_packing(downloaded, result, 1, opt);
        if (result.empty() || result.w != elems || result.elempack != 1)
        {
            fprintf(stderr, "FATAL: unpack produced empty/wrong result (w=%d elempack=%d)\n", result.w, result.elempack);
            delete op;
            vkdev->reclaim_staging_allocator(staging_vkallocator);
            vkdev->reclaim_blob_allocator(blob_vkallocator);
            ncnn::destroy_gpu_instance();
            return 1;
        }

        const float* q = result;
        float gpu_sum = 0.f;
        int mismatches = 0;
        for (int i = 0; i < elems; i++)
        {
            const float expect = (i % 2 == 0) ? float(i) : 0.f;
            const float got = q[i];
            if (expect != got)
            {
                if (mismatches < 8)
                    fprintf(stderr, "MISMATCH at %d: expected %.4f got %.4f\n", i, expect, got);
                mismatches++;
            }
            gpu_sum += got;
        }

        printf("gpu output: sum = %.1f (expected %.1f), mismatches = %d\n", gpu_sum, expected_sum, mismatches);
        printf("gpu output sample: [0]=%.2f [1]=%.2f [2]=%.2f ... [%d]=%.2f\n",
                q[0], q[1], q[2], elems - 1, q[elems - 1]);

        if (mismatches != 0 || gpu_sum != expected_sum)
        {
            fprintf(stderr, "FATAL: gpu result does not match cpu reference\n");
            delete op;
            vkdev->reclaim_staging_allocator(staging_vkallocator);
            vkdev->reclaim_blob_allocator(blob_vkallocator);
            ncnn::destroy_gpu_instance();
            return 1;
        }
    }

    printf("VULKAN COMPUTE OK on device %d (\"%s\")\n", pinned_index, pinned_info.device_name());

    // teardown: this is the only device we touched, so this only destroys our card
    op->destroy_pipeline(opt);
    delete op;
    vkdev->reclaim_staging_allocator(staging_vkallocator);
    vkdev->reclaim_blob_allocator(blob_vkallocator);

    ncnn::destroy_gpu_instance();

    printf("PROBE SUCCESS\n");
    return 0;
}
