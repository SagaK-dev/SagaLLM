"""Canonical metadata-only fingerprints of verified local checkpoint tensors."""
import hashlib
from pathlib import Path
import torch
from sagallm.exposure_trace import canonical, digest, file_sha, require

def tensor_record(t):
    require(isinstance(t,torch.Tensor) and t.device.type=="cpu" and t.layout==torch.strided
            and not t.is_quantized and t.numel()<=8_000_000,"Unsupported tensor")
    raw=t.detach().contiguous().reshape(-1).view(torch.uint8).numpy().tobytes()
    require(len(raw)==t.numel()*t.element_size(),"Tensor storage size mismatch")
    return {"dtype":str(t.dtype),"shape":list(t.shape),"bytes":len(raw),"sha256":digest(raw)}

def checkpoint_index(path,expected):
    p=Path(path)
    require(p.is_file() and not p.is_symlink() and 0<p.stat().st_size<=90*1024*1024
            and file_sha(p)==expected,"Checkpoint pin mismatch")
    require(torch.serialization.get_unsafe_globals_in_checkpoint(p)==[],"Unsupported globals")
    state=torch.load(p,map_location="cpu",weights_only=True)
    require(type(state) is dict and isinstance(state["model"],dict),"Checkpoint schema")
    # Contiguous byte view isolates logical tensor bytes; no values are exported.
    model={k:tensor_record(v) for k,v in sorted(state["model"].items())}
    return {"checkpoint_sha256":expected,"step":state["step"],"tensor_count":len(model),
            "tensor_bytes":sum(v["bytes"] for v in model.values()),
            "model_index_sha256":digest(canonical(model)),"config_sha256":digest(canonical(state["config"])),
            "tensors":model,"optimizer_state_indexed":False}
