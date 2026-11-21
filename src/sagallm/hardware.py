"""Explicit, local-only hardware diagnostics and bounded runtime smoke tests.

Nothing here provisions a GPU or starts multi-GPU training. CPU and GPU tests
are labeled separately: a successful runtime probe is not physical attestation.
"""
from __future__ import annotations

from datetime import datetime, timezone, timedelta
import importlib.util
import json
import math
import os
from pathlib import Path
import platform
import shutil
import subprocess
import sys
import tempfile
import time
from typing import Any

import torch
from torch import distributed as dist

from .config import ModelConfig
from .model import SagaTransformer
from .training import checkpoint_load, checkpoint_save, resolve_device

SCHEMA = "sagallm-hardware-v1"


def _git_sha() -> str | None:
    try:
        root = Path(__file__).resolve().parents[2]
        result = subprocess.run(
            ["git", "-C", str(root), "rev-parse", "HEAD"],
            capture_output=True, text=True, timeout=3, check=True,
        )
        value = result.stdout.strip()
        return value if len(value) == 40 and all(c in "0123456789abcdef" for c in value) else None
    except (OSError, subprocess.CalledProcessError, subprocess.TimeoutExpired):
        return None


def _available_ram() -> int | None:
    try:
        import psutil
        return int(psutil.virtual_memory().available)
    except (ImportError, OSError):
        return None


def _cuda_driver() -> str | None:
    # nvidia-smi is informative only. It is never required for a test to pass.
    try:
        result = subprocess.run(
            ["nvidia-smi", "--query-gpu=driver_version", "--format=csv,noheader"],
            capture_output=True, text=True, timeout=3, check=True,
        )
        return result.stdout.splitlines()[0].strip() or None
    except (OSError, IndexError, subprocess.CalledProcessError, subprocess.TimeoutExpired):
        return None


def inventory() -> dict[str, Any]:
    """Collect capabilities without allocating model weights or launching jobs."""
    cuda = bool(torch.cuda.is_available())
    mps = bool(hasattr(torch.backends, "mps") and torch.backends.mps.is_available())
    devices: list[dict[str, Any]] = []
    driver = _cuda_driver() if cuda else None
    if cuda:
        for i in range(torch.cuda.device_count()):
            properties = torch.cuda.get_device_properties(i)
            devices.append({
                "kind": "cuda", "index": i, "name": properties.name,
                "total_memory_bytes": int(properties.total_memory),
                "compute_capability": list(torch.cuda.get_device_capability(i)),
                "driver_version": driver,
                "pytorch_cuda_runtime": torch.version.cuda,
            })
    if mps:
        devices.append({
            "kind": "mps", "index": 0,
            "name": "Apple Metal Performance Shaders (device details unavailable)",
            "total_memory_bytes": None,
        })
    distributed = {
        "available": bool(dist.is_available()),
        "gloo_available": bool(dist.is_available() and dist.is_gloo_available()),
        "nccl_available": bool(dist.is_available() and dist.is_nccl_available()),
        "torchrun_module_present": importlib.util.find_spec("torch.distributed.run") is not None
        if dist.is_available() else False,
        "note": "Backend availability does not demonstrate multi-process or multi-GPU correctness.",
    }
    return {
        "schema": SCHEMA,
        "recorded_at_utc": datetime.now(timezone.utc).isoformat(),
        "source_sha": _git_sha(),
        "system": {
            "os": platform.platform(), "python": platform.python_version(),
            "torch": str(torch.__version__), "cpu_count": os.cpu_count(),
            "available_ram_bytes": _available_ram(),
            "free_disk_bytes": shutil.disk_usage(Path.cwd()).free,
        },
        "accelerators": devices,
        "default_device": str(resolve_device("auto")),
        "distributed": distributed,
        "status": "INVENTORY_ONLY",
        "physical_evidence": "NOT_ATTESTED",
    }


def _tiny_config() -> ModelConfig:
    return ModelConfig(vocab_size=128, hidden_size=48, num_layers=2, num_heads=4,
                       num_kv_heads=2, intermediate_size=96, max_position_embeddings=32)


def smoke(*, device_name: str = "cpu", precision: str = "fp32") -> dict[str, Any]:
    """Run exactly one tiny training step, checkpoint roundtrip and generation.

    Runs only on the explicitly selected local device. No training data is
    downloaded and no persistent model artifacts are created.
    """
    if precision not in {"fp32", "bf16"}:
        raise ValueError("precision must be fp32 or bf16")
    if device_name not in {"cpu", "cuda", "mps", "auto"}:
        raise ValueError("device must be cpu, cuda, mps or auto")
    try:
        device = resolve_device(device_name)
    except (ValueError, RuntimeError) as exc:
        return {"status": "NOT_RUN", "device_requested": device_name,
                "reason": str(exc), "physical_evidence": "NOT_ATTESTED"}
    if precision == "bf16" and not (
        device.type == "cuda" and torch.cuda.is_bf16_supported()
    ):
        return {"status": "NOT_RUN", "device_requested": device_name,
                "device": str(device), "precision": precision,
                "reason": "bf16 smoke is supported only on compatible CUDA devices",
                "physical_evidence": "NOT_ATTESTED"}

    start = time.monotonic()
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)
    # A local forked seed avoids changing the caller's global RNG state.
    with torch.random.fork_rng(devices=[device.index if device.index is not None else torch.cuda.current_device()]
                               if device.type == "cuda" else []):
        torch.manual_seed(20261121)
        config = _tiny_config()
        model = SagaTransformer(config).to(device)
        optimizer = torch.optim.AdamW(model.parameters(), lr=0.003)
        x = torch.randint(5, config.vocab_size, (1, 12), generator=torch.Generator().manual_seed(37)).to(device)
        y = x.roll(-1, dims=1)
        model.train()
        optimizer.zero_grad(set_to_none=True)
        amp = torch.autocast(device_type="cuda", dtype=torch.bfloat16) if precision == "bf16" else None
        if amp is not None:
            with amp:
                result = model(x, labels=y)
        else:
            result = model(x, labels=y)
        if result.loss is None or not torch.isfinite(result.loss).item():
            raise RuntimeError("Non-finite loss in hardware smoke")
        before_loss = float(result.loss.item())
        result.loss.backward()
        if not torch.isfinite(model.embedding.weight.grad).all().item():
            raise RuntimeError("Non-finite gradients in hardware smoke")
        old_weight = model.embedding.weight.detach().clone()
        optimizer.step()
        weights_changed = not torch.equal(old_weight, model.embedding.weight)
        if not weights_changed:
            raise RuntimeError("Optimizer did not update model parameters")
        with tempfile.TemporaryDirectory(prefix="sagallm-smoke-") as tmp:
            path = Path(tmp) / "checkpoint.pt"
            checkpoint_save(path, model, optimizer, 1, config, {"test_only": True})
            restored = SagaTransformer(config).to(device)
            restored_optimizer = torch.optim.AdamW(restored.parameters(), lr=0.003)
            loaded = checkpoint_load(path, restored, restored_optimizer)
            if loaded["step"] != 1:
                raise RuntimeError("Checkpoint step mismatch")
            checkpoint_match = all(torch.equal(a, b) for a, b in
                                   zip(model.state_dict().values(), restored.state_dict().values()))
            if not checkpoint_match:
                raise RuntimeError("Checkpoint model weights differ after reload")
        restored.eval()
        with torch.inference_mode():
            generated = restored.generate(x[:, :4], max_new_tokens=2, temperature=0)
        if generated.shape != (1, 6):
            raise RuntimeError("Text generation smoke yielded incorrect shape")
    if device.type == "cuda":
        torch.cuda.synchronize(device)
    return {
        "status": "PASS_RUNTIME", "device": str(device), "device_type": device.type,
        "precision": precision, "parameters": config.parameter_count(),
        "training_steps": 1, "loss": before_loss,
        "finite_gradients": True, "optimizer_updated": weights_changed,
        "checkpoint_roundtrip": checkpoint_match, "generated_tokens": 2,
        "elapsed_seconds": round(time.monotonic() - start, 3),
        "cuda_peak_allocated_bytes": int(torch.cuda.max_memory_allocated(device))
        if device.type == "cuda" else None,
        "physical_evidence": "NOT_ATTESTED",
    }


def _distributed_worker(output_path: Path) -> None:
    """Run under torchrun; no model or gradient synchronization is attempted."""
    dist.init_process_group("gloo", timeout=timedelta(seconds=15))
    try:
        rank, world = dist.get_rank(), dist.get_world_size()
        values = torch.tensor([rank + 1], dtype=torch.int64)
        dist.all_reduce(values, op=dist.ReduceOp.SUM)
        if world != 2 or int(values.item()) != 3:
            raise RuntimeError("Gloo collective produced an unexpected result")
        if rank == 0:
            output_path.write_text(json.dumps({"world_size": world, "all_reduce_sum": int(values.item())}))
        dist.barrier()
    finally:
        dist.destroy_process_group()


def distributed_smoke(*, timeout_seconds: int = 40) -> dict[str, Any]:
    """Explicitly test two local CPU/Gloo ranks; never run distributed training."""
    if timeout_seconds < 10 or timeout_seconds > 120:
        raise ValueError("timeout_seconds must be between 10 and 120")
    if not dist.is_available() or not dist.is_gloo_available():
        return {"status": "NOT_RUN", "reason": "Gloo backend is unavailable", "physical_evidence": "NOT_ATTESTED"}
    with tempfile.TemporaryDirectory(prefix="sagallm-gloo-") as temporary:
        output = Path(temporary) / "result.json"
        cmd = [sys.executable, "-m", "torch.distributed.run", "--standalone",
               "--nnodes=1", "--nproc-per-node=2", "-m", "sagallm.hardware",
               "_worker", str(output)]
        env = os.environ.copy()
        env.update({"CUDA_VISIBLE_DEVICES": "", "OMP_NUM_THREADS": "1", "MKL_NUM_THREADS": "1"})
        start = time.monotonic()
        try:
            child = subprocess.run(cmd, capture_output=True, text=True, env=env, timeout=timeout_seconds)
        except subprocess.TimeoutExpired:
            return {"status": "FAIL", "reason": "Local Gloo smoke timed out", "physical_evidence": "NOT_ATTESTED"}
        if child.returncode != 0 or not output.is_file():
            # Intentionally avoid copying arbitrary runtime stdout/stderr to a report.
            return {"status": "FAIL", "reason": "Two-rank Gloo smoke exited unsuccessfully",
                    "exit_code": child.returncode, "physical_evidence": "NOT_ATTESTED"}
        data = json.loads(output.read_text(encoding="utf-8"))
        if data != {"world_size": 2, "all_reduce_sum": 3}:
            return {"status": "FAIL", "reason": "Unexpected Gloo collective output", "physical_evidence": "NOT_ATTESTED"}
        return {"status": "PASS_RUNTIME", "backend": "gloo", "device": "cpu",
                "local_processes": 2, "all_reduce_sum": 3,
                "elapsed_seconds": round(time.monotonic() - start, 3),
                "multi_gpu_training": "NOT_RUN", "physical_evidence": "NOT_ATTESTED"}


def save_report(path: str | Path, payload: dict[str, Any]) -> Path:
    """Write a new evidence JSON file; never overwrite prior runs."""
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    with target.open("x", encoding="utf-8") as handle:
        json.dump(payload, handle, ensure_ascii=False, indent=2, allow_nan=False)
        handle.write("\n")
    return target


def run_hardware_command(action: str, *, device: str = "cpu", precision: str = "fp32",
                         timeout: int = 40, output: str | None = None) -> dict[str, Any]:
    if output is not None and Path(output).exists():
        raise FileExistsError(f"Refusing to overwrite existing hardware evidence: {output}")
    report = inventory()
    if action == "smoke":
        report["runtime_test"] = smoke(device_name=device, precision=precision)
    elif action == "distributed-smoke":
        report["runtime_test"] = distributed_smoke(timeout_seconds=timeout)
    elif action != "inspect":
        raise ValueError(f"Unknown hardware command: {action}")
    if "runtime_test" in report:
        check = report["runtime_test"]["status"]
        report["status"] = "RUNTIME_TESTED" if check == "PASS_RUNTIME" else check
    if output is not None:
        save_report(output, report)
        report["report_path"] = output
    return report


def main() -> None:
    # Private subprocess entrypoint, used only by the explicitly triggered smoke.
    if len(sys.argv) != 3 or sys.argv[1] != "_worker":
        raise SystemExit("This module's worker entrypoint is internal. Use `sagallm hardware`." )
    _distributed_worker(Path(sys.argv[2]))


if __name__ == "__main__":
    main()