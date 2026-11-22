"""Run opt-in, bounded CPU SFT starting from pinned v0.16 real-Java weights.

No model-generated Java is executed: generation is evaluated by javac only.
"""
from __future__ import annotations
import argparse
import json
from pathlib import Path

from sagallm.java_instruction_sft import train_instruction_sft


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
    a=p.parse_args()
    report=train_instruction_sft(a.base,a.tokenizer,a.dataset,a.output,steps=a.steps,
         batch_size=a.batch_size,eval_every=a.eval_every,max_seconds=a.max_seconds,
         compile_cases=a.compile_cases,lr=a.lr)
    print('JAVA_INSTRUCTION_SFT_SUMMARY '+json.dumps({k:v for k,v in report.items() if k!='history'},ensure_ascii=False),flush=True)


if __name__=='__main__':
    main()