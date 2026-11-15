"""Query-time benchmark for RaP-Weather (Table 2). Baseline latencies are
measured separately."""
import json
import os
import platform
import time

import numpy as np
import torch

from .config import CONFIG, device, run_dir
from .models import cleanup_gpu, load_trained_model


def measure_inference_speed(model, device, input_shape=(1, 3, 224, 224), iterations=100):
    model.eval()
    dummy_input = torch.randn(input_shape).to(device)
    with torch.no_grad():
        for _ in range(10):
            _ = model.forward_spatial(dummy_input)
    if device.type == 'cuda': torch.cuda.synchronize()

    timings = []
    with torch.no_grad():
        for _ in range(iterations):
            if device.type == 'cuda':
                torch.cuda.synchronize(); start = time.time()
                _ = model.forward_spatial(dummy_input)
                torch.cuda.synchronize(); end = time.time()
            else:
                start = time.time(); _ = model.forward_spatial(dummy_input); end = time.time()
            timings.append((end - start) * 1000)

    avg_time, std_time = np.mean(timings), np.std(timings)
    hw_info = {
        "gpu": torch.cuda.get_device_name(0) if torch.cuda.is_available() else "CPU only",
        "cuda_version": torch.version.cuda, "torch_version": torch.__version__,
        "cpu": platform.processor() or "unavailable", "os": platform.platform(),
    }
    print(f"Avg Latency: {avg_time:.4f} ms ± {std_time:.4f} ms (n={iterations}) | {1000/avg_time:.2f} FPS")
    print(f"   Hardware: {hw_info['gpu']} | CUDA {hw_info['cuda_version']} | torch {hw_info['torch_version']} | CPU: {hw_info['cpu']}")
    return {"avg_ms": avg_time, "std_ms": std_time, "fps": 1000 / avg_time, "hardware": hw_info}


def main(cfg=CONFIG):
    speed_model = load_trained_model(cfg=cfg, device=device)
    n_speed_runs = 5
    speed_runs = [measure_inference_speed(speed_model, device) for _ in range(n_speed_runs)]
    avg_across_runs = np.mean([r["avg_ms"] for r in speed_runs])
    std_across_runs = np.std([r["avg_ms"] for r in speed_runs], ddof=1)
    print(f"\nAcross {n_speed_runs} independent runs: {avg_across_runs:.4f} ms ± {std_across_runs:.4f} ms")
    print(f"Hardware for Table 2: {speed_runs[0]['hardware']}")

    speed_report = {"avg_ms": avg_across_runs, "std_ms": std_across_runs, "n_runs": n_speed_runs, "hardware": speed_runs[0]["hardware"]}
    with open(os.path.join(run_dir(cfg), "query_time_rap_weather.json"), "w") as f:
        json.dump(speed_report, f, indent=2, default=str)
    del speed_model
    cleanup_gpu("speed_model")
    return speed_report


if __name__ == "__main__":
    main()
