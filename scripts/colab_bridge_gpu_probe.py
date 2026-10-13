"""Small CUDA/CPU sanity check. No model training or datasets."""
import json
import platform

result = {"status": "ok", "cuda_available": False, "gpu": "CPU"}
try:
    import torch
    result["torch"] = str(torch.__version__)
    result["cuda_available"] = bool(torch.cuda.is_available())
    if torch.cuda.is_available():
        result["gpu"] = torch.cuda.get_device_name(0)
        a = torch.eye(8, device="cuda")
    else:
        a = torch.eye(8)
    assert torch.equal(a @ a, a)
except ImportError:
    result["status"] = "torch-missing"
    result["note"] = f"Python {platform.python_version()}"

print("COLAB_PUBLIC_RESULT=" + json.dumps(result, sort_keys=True), flush=True)
