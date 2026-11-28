"""Prospective CPU SFT exposure journal; no historical exposure certification.

The journal records packed input and label digests before forward, and records
successful optimizer returns separately. It is a local hash chain, not an
authenticated attestation or proof of gradient influence.
"""
from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
import re
import struct

import torch
from .sft_lr_schedule import validate_schedule, step_lr, group_lr

SCHEMA = "sagallm-optimizer-exposure-v1"
ZERO = "0" * 64
MAX_STEPS = 1500
MAX_ROWS = 8
MAX_SEQUENCE = 4096
MAX_EVENTS = MAX_STEPS * 3 + 4
MAX_BYTES = 24 * 1024 * 1024
HEX = re.compile(r"[0-9a-f]{64}\Z")


def require(ok, message):
    if not ok:
        raise ValueError(message)


def digest(raw):
    return hashlib.sha256(raw).hexdigest()


def canonical(value):
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False,
                      allow_nan=False).encode("utf8")


def is_sha(v):
    return isinstance(v, str) and HEX.fullmatch(v) is not None


def file_sha(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as f:
        for block in iter(lambda: f.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def tensor_sha(t):
    require(isinstance(t, torch.Tensor) and t.dtype == torch.long and t.device.type == "cpu"
            and t.ndim in (1, 2) and 0 < t.numel() <= MAX_ROWS * MAX_SEQUENCE,
            "Unsupported CPU SFT token tensor")
    # Fixed little-endian int64 encoding, including dimension/shape.
    shape = tuple(t.shape)
    values = t.detach().contiguous().reshape(-1).tolist()
    raw = struct.pack("<I", len(shape)) + struct.pack("<" + "I"*len(shape), *shape)
    raw += struct.pack("<" + "q"*len(values), *values)
    return digest(b"sagallm-token-tensor-v1\0" + raw)


def batch_record(rows, x, y):
    require(isinstance(rows, (list, tuple)) and 1 <= len(rows) <= MAX_ROWS
            and isinstance(x, torch.Tensor) and isinstance(y, torch.Tensor)
            and x.ndim == y.ndim == 2 and x.shape == y.shape
            and x.shape[0] == len(rows) and 0 < x.shape[1] <= MAX_SEQUENCE,
            "Invalid packed batch")
    xs, ys = tensor_sha(x), tensor_sha(y)
    result = []
    for i, row in enumerate(rows):
        rx, ry = row["x"], row["y"]
        require(rx.ndim == ry.ndim == 1 and rx.shape == ry.shape
                and 0 < rx.numel() <= x.shape[1], "Invalid encoded row")
        rxh, ryh = tensor_sha(rx), tensor_sha(ry)
        n = rx.numel()
        require(torch.equal(x[i,:n], rx) and torch.equal(y[i,:n], ry)
                and bool((x[i,n:] == 0).all()) and bool((y[i,n:] == -100).all()),
                "Packed row or padding differs from encoded source")
        require(bool((rx >= 0).all()) and bool(((ry >= 0) | (ry == -100)).all()),
                "Invalid token labels")
        tokens = int((ry != -100).sum())
        require(tokens > 0 and tokens == row["response_tokens"], "Target count mismatch")
        ident = row["id"]
        family = row["family"]
        require(isinstance(ident,str) and 0 < len(ident) <= 512 and isinstance(family,str)
                and 0 < len(family) <= 512, "Invalid row identity")
        result.append({"id_sha256":digest(b"row-id-v1\0"+ident.encode()),
                       "family_sha256":digest(b"family-v1\0"+family.encode()),
                       "input_sha256":rxh,"labels_sha256":ryh,
                       "positions":n,"loss_targets":tokens})
    return {"rows":result,"input_sha256":xs,"labels_sha256":ys,
            "shape":list(x.shape),"loss_targets":sum(r["loss_targets"] for r in result)}


def validate_header(header):
    allowed={"schema","split","base_checkpoint_sha256","tokenizer_sha256",
             "data_manifest_sha256","train_split_sha256","seed","max_steps","batch_size"}
    require(set(header)-{"lr_schedule"}==allowed
            and header["schema"]==SCHEMA and header["split"]=="train"
            and all(is_sha(header[k]) for k in ("base_checkpoint_sha256","tokenizer_sha256",
                                              "data_manifest_sha256","train_split_sha256"))
            and type(header["seed"]) is int and 0<=header["seed"]<2**63
            and type(header["max_steps"]) is int and 1<=header["max_steps"]<=MAX_STEPS
            and type(header["batch_size"]) is int and 1<=header["batch_size"]<=MAX_ROWS,
            "Invalid trace header")
    if "lr_schedule" in header:
        validate_schedule(header["lr_schedule"],header["max_steps"])


class ExposureTrace:
    def __init__(self,path,header):
        validate_header(header)
        self.path=Path(path)
        self.header=dict(header)
        self.tip=ZERO
        self.events=0
        self.completed=0
        self.pending=None
        self.finished=False
        self.poisoned=False
        self.checkpoints={}
        # Exclusive creation; no resume or overwrite in v1.
        self.handle=self.path.open("xb")
        try:
            self._append("header",self.header)
        except BaseException:
            self.handle.close()
            raise

    def __enter__(self):
        return self

    def __exit__(self,*exc):
        self.handle.close()

    def _append(self,kind,payload):
        require(not self.finished and not self.poisoned and self.events<MAX_EVENTS,"Trace closed or poisoned")
        body={"seq":self.events,"prev_sha256":self.tip,"kind":kind,"payload":payload}
        h=digest(canonical(body))
        raw=canonical({**body,"event_sha256":h})+b"\n"
        require(self.handle.tell()+len(raw)<=MAX_BYTES,"Trace size limit")
        try:
            self.handle.write(raw)
            self.handle.flush()
            os.fsync(self.handle.fileno())
        except BaseException:
            self.poisoned=True
            raise
        self.tip=h
        self.events+=1
        return h

    def _begin(self,rows,x,y,optimizer_lr=None):
        require(self.pending is None and self.completed<self.header["max_steps"]
                and "last" not in self.checkpoints
                and len(rows)==self.header["batch_size"],"Unexpected training step")
        record=batch_record(rows,x,y)
        step=self.completed+1
        if "lr_schedule" in self.header:
            require(type(optimizer_lr) is float and optimizer_lr==step_lr(self.header["lr_schedule"],self.header["max_steps"],step),
                    "Actual optimizer learning rate differs from schedule")
            record["optimizer_lr"]=optimizer_lr
        else:
            require(optimizer_lr is None,"Unexpected optimizer learning rate")
        intent=self._append("step_intent",{"step":step,**record})
        self.pending=(step,intent)
        return intent

    def _applied(self,intent):
        require(self.pending==(self.completed+1,intent),"Unmatched optimizer return")
        step=self.completed+1
        self._append("optimizer_returned",{"step":step,"intent_sha256":intent})
        self.completed=step
        self.pending=None

    def checkpoint_metadata(self):
        require(self.pending is None and not self.finished and not self.poisoned,"Unresolved trace state")
        return {"schema":SCHEMA,"completed_optimizer_returns":self.completed,"prefix_sha256":self.tip}

    def bind_checkpoint(self,role,path,metadata):
        require(role in ("best","last") and "last" not in self.checkpoints
                and (role!="last" or "best" in self.checkpoints)
                and (role!="best" or "best" not in self.checkpoints or
                     self.checkpoints["best"]["completed_optimizer_returns"]<self.completed)
                and metadata==self.checkpoint_metadata(),"Invalid checkpoint binding")
        value={"role":role,"checkpoint_sha256":file_sha(path),**metadata}
        self._append("checkpoint",value)
        self.checkpoints[role]=value
        return value

    def finish(self):
        require(self.pending is None and set(self.checkpoints)=={"best","last"}
                and self.checkpoints["last"]["completed_optimizer_returns"]==self.completed,
                "Cannot finalize incomplete training/checkpoint trace")
        self._append("final",{"completed_optimizer_returns":self.completed,
                              "checkpoint_sha256":{k:v["checkpoint_sha256"] for k,v in self.checkpoints.items()}})
        self.finished=True
        return {"trace_sha256":file_sha(self.path),"tip_sha256":self.tip,
                "completed_optimizer_returns":self.completed,"events":self.events}


def traced_sft_step(model,optimizer,trace,rows,x,y):
    """Actual forward/backward/optimizer path; a failed step stays incomplete."""
    actual_lr=group_lr(optimizer) if "lr_schedule" in trace.header else None
    intent=trace._begin(rows,x,y,actual_lr)
    try:
        optimizer.zero_grad(set_to_none=True)
        loss=model(x,labels=y).loss
        if not bool(torch.isfinite(loss)):
            raise FloatingPointError("Nonfinite SFT loss")
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(),1.0,error_if_nonfinite=True)
        optimizer.step()
        trace._applied(intent)
        return float(loss.detach())
    except BaseException:
        trace.poisoned=True
        raise


def _unique(pairs):
    out={}
    for k,v in pairs:
        require(k not in out,"Duplicate JSON key")
        out[k]=v
    return out


def verify_trace(path,*,expected_sha256=None,expected_tip=None):
    p=Path(path)
    require(p.is_file() and not p.is_symlink() and 0<p.stat().st_size<=MAX_BYTES,"Unsafe trace path")
    raw=p.read_bytes()
    require(raw.endswith(b"\n") and len(raw)<=MAX_BYTES,"Truncated or oversized trace")
    h=digest(raw)
    if expected_sha256 is not None:
        require(h==expected_sha256,"Trace file digest mismatch")
    lines=raw.splitlines()
    require(1<=len(lines)<=MAX_EVENTS,"Trace event count")
    tip=ZERO; header=None; pending=None; completed=0; checkpoints={}; finished=False
    for seq,line in enumerate(lines):
        row=json.loads(line,object_pairs_hook=_unique)
        require(set(row)=={"seq","prev_sha256","kind","payload","event_sha256"} and type(row["seq"]) is int
                and row["seq"]==seq and row["prev_sha256"]==tip and not finished,"Trace order mismatch")
        event=row.pop("event_sha256")
        require(event==digest(canonical(row)) and line==canonical({**row,"event_sha256":event}),"Trace event hash mismatch")
        kind,pay=row["kind"],row["payload"]
        if seq==0:
            require(kind=="header","Missing header");validate_header(pay);header=pay
        elif kind=="step_intent":
            expected={"step","rows","input_sha256","labels_sha256","shape","loss_targets"}
            if "lr_schedule" in header:
                expected.add("optimizer_lr")
                require(type(pay.get("optimizer_lr")) is float and pay["optimizer_lr"]==step_lr(header["lr_schedule"],header["max_steps"],completed+1),"Trace learning rate mismatch")
            require(set(pay)==expected
                    and pending is None and "last" not in checkpoints and type(pay["step"]) is int
                    and pay["step"]==completed+1<=header["max_steps"]
                    and len(pay["rows"])==header["batch_size"]
                    and is_sha(pay["input_sha256"]) and is_sha(pay["labels_sha256"])
                    and len(pay["shape"])==2 and pay["shape"][0]==header["batch_size"]
                    and type(pay["shape"][1]) is int and 0<pay["shape"][1]<=MAX_SEQUENCE,
                    "Invalid step intent")
            for r in pay["rows"]:
                require(set(r)=={"id_sha256","family_sha256","input_sha256","labels_sha256","positions","loss_targets"}
                        and all(is_sha(r[k]) for k in ("id_sha256","family_sha256","input_sha256","labels_sha256"))
                        and type(r["positions"]) is int and type(r["loss_targets"]) is int
                        and 0<r["loss_targets"]<=r["positions"]<=pay["shape"][1],"Invalid row digest record")
            require(pay["loss_targets"]==sum(r["loss_targets"] for r in pay["rows"]),"Loss-target count drift")
            pending=(pay["step"],event)
        elif kind=="optimizer_returned":
            require(set(pay)=={"step","intent_sha256"} and pending==(pay["step"],pay["intent_sha256"]),
                    "Optimizer return without matching intent")
            completed+=1;pending=None
        elif kind=="checkpoint":
            require(set(pay)=={"role","checkpoint_sha256","schema","completed_optimizer_returns","prefix_sha256"}
                    and pending is None and "last" not in checkpoints and pay["role"] in ("best","last")
                    and (pay["role"]!="last" or "best" in checkpoints)
                    and (pay["role"]!="best" or "best" not in checkpoints or
                         checkpoints["best"]["completed_optimizer_returns"]<completed)
                    and is_sha(pay["checkpoint_sha256"])
                    and pay["schema"]==SCHEMA and pay["completed_optimizer_returns"]==completed
                    and pay["prefix_sha256"]==tip,"Invalid checkpoint event")
            checkpoints[pay["role"]]=pay
        elif kind=="final":
            require(pending is None and set(checkpoints)=={"best","last"}
                    and checkpoints["last"]["completed_optimizer_returns"]==completed
                    and pay=={"completed_optimizer_returns":completed,"checkpoint_sha256":{
                        k:v["checkpoint_sha256"] for k,v in checkpoints.items()}},"Invalid final event")
            finished=True
        else:
            raise ValueError("Unexpected trace event")
        tip=event
    if expected_tip is not None:
        require(tip==expected_tip,"Trace tip mismatch")
    return {"status":"FINALIZED_LOCAL_TRACE" if finished else "INCOMPLETE_LOCAL_TRACE",
            "trace_sha256":h,"tip_sha256":tip,"events":len(lines),"completed_optimizer_returns":completed,
            "pending_step":None if pending is None else pending[0],"checkpoints":checkpoints,
            "authenticated_attestation":False,"gradient_influence_verified":False,
            "historical_exposure_verified":False}
