"""Run opt-in, bounded CPU SFT starting from pinned v0.16 real-Java weights.

No model-generated Java is executed: generation is evaluated by javac only.
"""
from __future__ import annotations
import argparse
import json
from pathlib import Path

from sagallm.java_instruction_sft import train_instruction_sft, V016_BASE_BEST_SHA256, V017_SFT_BEST_SHA256


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--base',type=Path,required=True)
    p.add_argument('--tokenizer',type=Path,required=True)
    p.add_argument('--dataset',type=Path,required=True)
    p.add_argument('--output',type=Path,required=True)
    p.add_argument('--steps',type=int,default=500)
    p.add_argument('--batch-size',type=int,default=4)
    p.add_argument('--eval-every',type=int,default=100)
    p.add_argument('--max-seconds',type=float,default=360)
    p.add_argument('--compile-cases',type=int,default=10)
    p.add_argument('--lr',type=float,default=0.00015)
    p.add_argument('--context',type=int,choices=[384],help='Opt-in 192→384 RoPE extrapolation using pinned v0.17 SFT weights')
    p.add_argument('--record-exposure',action='store_true',help='Record and verify a local optimizer-input journal; keep journal private until reviewed')
    p.add_argument('--validation-only',action='store_true',help='Skip TEST scoring; requires --compile-cases 0')
    p.add_argument('--lr-schedule',choices=['constant','warmup_cosine'])
    p.add_argument('--warmup-steps',type=int,default=0)
    p.add_argument('--min-lr-ratio',type=float,default=1.0)
    p.add_argument('--dataset-mode',choices=['canonical','template_capped_v1'],default='canonical')
    a=p.parse_args()
    if not a.lr_schedule and (a.warmup_steps!=0 or a.min_lr_ratio!=1.0):
        p.error('Schedule parameters require --lr-schedule')
    schedule=None if not a.lr_schedule else {'kind':a.lr_schedule,'peak_lr':a.lr,
                 'warmup_steps':a.warmup_steps,'min_lr_ratio':a.min_lr_ratio}
    report=train_instruction_sft(a.base,a.tokenizer,a.dataset,a.output,steps=a.steps,
         batch_size=a.batch_size,eval_every=a.eval_every,max_seconds=a.max_seconds,
         compile_cases=a.compile_cases,lr=a.lr,
         target_context=a.context,record_exposure=a.record_exposure,validation_only=a.validation_only,lr_schedule=schedule,dataset_mode=a.dataset_mode,
         pinned_checkpoint_sha256=V017_SFT_BEST_SHA256 if a.context else V016_BASE_BEST_SHA256)
    print('JAVA_INSTRUCTION_SFT_SUMMARY '+json.dumps({k:v for k,v in report.items() if k!='history'},ensure_ascii=False),flush=True)


if __name__=='__main__':
    main()