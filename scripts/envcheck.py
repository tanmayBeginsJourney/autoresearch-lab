"""GPU environment validation, run inside a Slurm allocation. Prints one JSON object."""
import json
import os
import time
import urllib.request

out = {"host": os.uname().nodename, "cuda_visible": os.environ.get("CUDA_VISIBLE_DEVICES")}
try:
    import torch
    out.update(torch=torch.__version__, cuda=torch.version.cuda, available=torch.cuda.is_available())
    if torch.cuda.is_available():
        p = torch.cuda.get_device_properties(0)
        out.update(gpu=p.name, cc=f"{p.major}.{p.minor}", mem_gb=round(p.total_memory / 2**30, 1), sms=p.multi_processor_count)
        a = torch.randn(8192, 8192, device="cuda", dtype=torch.bfloat16)
        for _ in range(3):
            a @ a
        torch.cuda.synchronize(); t = time.time(); n = 20
        for _ in range(n):
            a @ a
        torch.cuda.synchronize()
        out["bf16_matmul_tflops"] = round(2 * 8192**3 * n / (time.time() - t) / 1e12, 1)
        q = torch.randn(4, 16, 2048, 128, device="cuda", dtype=torch.bfloat16)
        from torch.nn.attention import SDPBackend, sdpa_kernel
        try:
            with sdpa_kernel(SDPBackend.FLASH_ATTENTION):
                torch.nn.functional.scaled_dot_product_attention(q, q, q, is_causal=True)
            out["flash_sdpa"] = True
        except RuntimeError as e:
            out["flash_sdpa"] = f"no: {str(e)[:80]}"
except Exception as e:  # report, do not crash: this is a diagnostic
    out["torch_error"] = f"{type(e).__name__}: {e}"[:300]
try:
    import triton
    import triton.language as tl

    @triton.jit
    def _add(x, y, o, n, B: tl.constexpr):
        i = tl.program_id(0) * B + tl.arange(0, B)
        m = i < n
        tl.store(o + i, tl.load(x + i, mask=m) + tl.load(y + i, mask=m), mask=m)

    x = torch.randn(4096, device="cuda"); o = torch.empty_like(x)
    _add[(4,)](x, x, o, 4096, B=1024)
    out["triton"] = f"{triton.__version__} ok={bool(torch.allclose(o, 2 * x))}"
except Exception as e:
    out["triton"] = f"error: {type(e).__name__}: {str(e)[:120]}"
try:
    urllib.request.urlopen("https://huggingface.co", timeout=8)
    out["compute_node_internet"] = True
except Exception as e:
    out["compute_node_internet"] = f"no ({type(e).__name__})"
print(json.dumps(out))
