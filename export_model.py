"""Export microsoft/unixcoder-base to ONNX for ONNX Runtime.
CPU-only: does NOT touch any GPU (neither the GTX 1050 nor the 1060).
Run once on any machine with `optimum` installed.
"""
from optimum.exporters.onnx import main_export
from optimum.exporters.tasks import TasksManager

model_id = "microsoft/unixcoder-base"   # 124M params, ~500MB fp32, fits 2GB VRAM
output_dir = "unixcoder-onnx"

# Use optimum.exporters.onnx.main_export to convert the model.
# It handles PyTorch -> ONNX conversion and saves the model.
# Device is implicitly CPU for export, as it does not initialize CUDA.
main_export(
    model_id=model_id,
    output=output_dir,
    task=TasksManager.infer_task_from_model_id(model_id),
    # CPUExecutionProvider is the default for export, so no need to specify
    # but being explicit is fine if desired: provider="CPUExecutionProvider"
)

print(f"Exported to {output_dir}")
