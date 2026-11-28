"""Bounded CPU SFT learning-rate schedule with exact, replayable step semantics."""
import math

def validate_schedule(spec, steps):
    if (not isinstance(spec,dict) or set(spec)!={"kind","peak_lr","warmup_steps","min_lr_ratio"}
        or type(steps) is not int or not 1<=steps<=1500):
        raise ValueError("Invalid schedule specification")
    kind=spec["kind"];peak=spec["peak_lr"];warm=spec["warmup_steps"];floor=spec["min_lr_ratio"]
    if (kind not in ("constant","warmup_cosine") or type(peak) not in (float,int)
        or not 0<peak<=.001 or type(warm) is not int or type(floor) not in (float,int)
        or not 0<floor<=1):
        raise ValueError("Invalid schedule parameters")
    if kind=="constant":
        if warm!=0 or floor!=1:raise ValueError("Constant schedule must use zero warmup and unit floor")
    elif not 1<=warm<steps-1:
        raise ValueError("Cosine schedule requires warmup and at least two decay updates")

def step_lr(spec,steps,step):
    validate_schedule(spec,steps)
    if type(step) is not int or not 1<=step<=steps:raise ValueError("Invalid optimizer step")
    peak=spec["peak_lr"];warm=spec["warmup_steps"]
    if spec["kind"]=="constant":return float(peak)
    if step<=warm:return peak*step/warm
    progress=(step-warm)/(steps-warm)
    floor=spec["min_lr_ratio"]
    return peak*(floor+(1-floor)*(1+math.cos(math.pi*progress))/2)

def group_lr(optimizer):
    rates=[g["lr"] for g in optimizer.param_groups]
    if not rates or any(type(r) not in (float,int) or not math.isfinite(r) or r<=0 for r in rates) or len(set(rates))!=1:
        raise ValueError("Expected a single finite positive learning rate across optimizer groups")
    return float(rates[0])
