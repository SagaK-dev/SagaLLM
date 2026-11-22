"""Train a fresh native SagaLLM on pinned, permissively-licensed Java source code.

This is a **code-language-modeling** experiment, not Java instruction following.
No Java code is executed, and model capabilities are not inferred from loss.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import random
import time

import torch

from sagallm.config import ModelConfig
from sagallm.java_public_corpus import verify_public_corpus, UPSTREAMS
from sagallm.model import SagaTransformer
from sagallm.tokenization import ByteBPETokenizer, FastBPETokenizer
from sagallm.training import checkpoint_save, checkpoint_load


def file_sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def tokens_for_file(tokenizer, text: str) -> list[int]:
    return ([tokenizer.SPECIAL['<bos>']] + tokenizer.encode(text)
            + [tokenizer.SPECIAL['<eos>']])


def make_windows(ids: list[int], block: int, limit: int) -> list[tuple[torch.Tensor, torch.Tensor]]:
    """Non-overlapping targets per file, EOS-inclusive, right-pad labels masked."""
    result=[]
    for start in range(0,len(ids)-1,block):
        if len(result) >= limit:break
        seq=ids[start:start+block+1]
        if len(seq)<2:break
        xx=seq[:-1]; yy=seq[1:]
        n=block-len(xx)
        result.append((torch.tensor(xx+[0]*n,dtype=torch.long),
                       torch.tensor(yy+[-100]*n,dtype=torch.long)))
    return result


def assign_windows(docs: list[dict],tokenizer,block:int,per_file:int):
    return [make_windows(tokens_for_file(tokenizer,d['text']),block,per_file) for d in docs]


def eval_windows(model, windows, limit:int=32):
    if not windows:return None
    was_training=model.training
    model.eval()
    losses=[]
    with torch.inference_mode():
        for x,y in windows[:limit]:
            loss=model(x.unsqueeze(0),labels=y.unsqueeze(0)).loss
            losses.append(float(loss))
    model.train(was_training)
    return sum(losses)/len(losses)


def balanced_first_windows(docs, window_sets, limit):
    """Choose held-out files round-robin by repository, never only first repo."""
    buckets={}
    for row, windows in zip(docs, window_sets):
        if windows:
            buckets.setdefault(row['repo'],[]).append(windows[0])
    ordered=[]
    for k in range(max(map(len,buckets.values()))):
        for name in sorted(buckets):
            if k<len(buckets[name]):
                ordered.append(buckets[name][k])
                if len(ordered)==limit:return ordered
    return ordered


def train(corpus:Path, output:Path, *, steps:int=950, max_seconds:float=480.0,
          batch_size:int=6, block_size:int=192, per_file:int=4,
          vocab_size:int=2048, backend:str='hf',
          lr:float=0.0003, seed:int=616, eval_every:int=200) -> dict:
    if not (1<=steps<=6000 and 5<=max_seconds<=900 and 1<=batch_size<=12
            and 32<=block_size<=256 and 1<=per_file<=12
            and 260<=vocab_size<=8192 and backend in {'hf','legacy'}
            and 0<lr<=0.01 and 10<=eval_every<=1500):
        raise ValueError('Training budget is outside safe CPU limits')
    if output.exists() or output.is_symlink():raise FileExistsError('Refuse output overwrite')
    integrity=verify_public_corpus(corpus)
    docs=integrity['files']
    for split in ('train','validation','test'):
        if not docs[split]:raise ValueError('Empty source split')
    torch.set_num_threads(min(2,max(1,os.cpu_count() or 1)))
    torch.manual_seed(seed)
    random.seed(seed)
    if backend=='hf':
        tokenizer=FastBPETokenizer.train((x['text'] for x in docs['train']),
                                         vocab_size=vocab_size,min_frequency=2)
    else:
        # Legacy Python BPE is too slow on multi-megabyte corpora; tiny offline smoke only.
        sample=[d['text'][:600] for d in docs['train'][:80]]
        tokenizer=ByteBPETokenizer.train(sample,vocab_size=min(vocab_size,512),min_frequency=2)
    cfg=ModelConfig(vocab_size=max(vocab_size,tokenizer.vocab_size),
                    hidden_size=192,num_layers=4,num_heads=4,num_kv_heads=2,
                    intermediate_size=384,max_position_embeddings=block_size)
    if cfg.parameter_count()>3_000_000:raise ValueError('Model allocation exceeds guard')
    train_windows=assign_windows(docs['train'],tokenizer,block_size,per_file)
    validation_windows=assign_windows(docs['validation'],tokenizer,block_size,1)
    validation_select=balanced_first_windows(docs['validation'],validation_windows,24)
    if not validation_select:raise ValueError('Missing validation windows')
    model=SagaTransformer(cfg)
    opt=torch.optim.AdamW(model.parameters(),lr=lr,weight_decay=0.01)
    initial_val=eval_windows(model,validation_select,24)
    output.mkdir(parents=True)
    tokenizer_path=output/'tokenizer.json';tokenizer.save(tokenizer_path)
    metadata={'stage':'pinned_public_java_lm','tokenizer_sha256':file_sha(tokenizer_path),
              'train_sha256':file_sha(corpus/'train.jsonl'),
              'dataset_manifest_sha256':file_sha(corpus/'manifest.json'),
              'seed':seed,'batch_size':batch_size,'block_size':block_size,
              'optimizer_base_lr':lr,'upstream_source_count':len(integrity['sources']),
              'source_license_status':'PINNED_TEXT_HASH_NOT_LEGAL_ATTESTATION'}
    best_path=output/'java-best.pt'
    checkpoint_save(best_path,model,opt,0,cfg,{**metadata,'selection_validation_loss':initial_val})
    # The first pass across all files is interleaved so a model stopped early
    # learns from many distinct files, not only a handful of long sources.
    rng=random.Random(seed)
    file_order=list(range(len(train_windows)));rng.shuffle(file_order)
    order=[]
    for part in range(per_file):
        for index in file_order:
            if part<len(train_windows[index]):order.append((index,part))
    if not order:raise ValueError('No usable Java code windows')
    start=time.monotonic();seen_files=set();seen_windows=set();best_val=initial_val
    best_step=0;best_files=0;best_windows=0;done=0;last_loss=None;history=[];termination='STEP_LIMIT'
    model.train()
    for step in range(1,steps+1):
        if time.monotonic()-start>max_seconds:
            termination='TIME_BUDGET';break
        selected=[]
        for k in range(batch_size):
            pos=(step-1)*batch_size+k
            idx,part=order[pos % len(order)]
            selected.append(train_windows[idx][part])
            seen_files.add(idx);seen_windows.add((idx,part))
        x=torch.stack([i[0] for i in selected]);y=torch.stack([i[1] for i in selected])
        opt.zero_grad(set_to_none=True)
        loss=model(x,labels=y).loss
        if not torch.isfinite(loss):raise FloatingPointError('Nonfinite loss')
        loss.backward();torch.nn.utils.clip_grad_norm_(model.parameters(),1.0)
        opt.step();done=step;last_loss=float(loss.detach())
        if step%eval_every==0 or step==steps:
            val=eval_windows(model,validation_select,24)
            history.append({'step':step,'loss_last_batch':last_loss,'validation_sample_loss':val,
                            'distinct_training_files_seen':len(seen_files),
                            'distinct_training_windows_seen':len(seen_windows)})
            print(json.dumps(history[-1]),flush=True)
            if val < best_val-0.001:
                best_val=val;best_step=step;best_files=len(seen_files);best_windows=len(seen_windows)
                checkpoint_save(best_path,model,opt,step,cfg,
                                {**metadata,'selection_validation_loss':val})
    checkpoint_save(output/'java-last.pt',model,opt,done,cfg,
                    {**metadata,'selection':'last','distinct_training_files_seen':len(seen_files)})
    checkpoint_load(best_path,model)
    model.eval()
    validation_full=balanced_first_windows(docs['validation'],validation_windows,64)
    test_windows=assign_windows(docs['test'],tokenizer,block_size,1)
    test_first=balanced_first_windows(docs['test'],test_windows,64)
    full_val=eval_windows(model,validation_full,64)
    test_loss=eval_windows(model,test_first,64)
    report={'status':'TRAINED_CPU','training_task':'JAVA_SOURCE_NEXT_TOKEN_ONLY',
            'javascript_or_java':'Java',
            'random_initialization':True,'external_pretrained_weights':False,
            'original_synthetic_sft_examples_used_in_this_run':0,
            'source_counts':integrity['counts'],'source_files_total':integrity['total'],
            'source_urls':[s.url for s in UPSTREAMS],
            'optimizer_steps_requested':steps,'optimizer_steps_completed':done,
            'distinct_training_files_seen':len(seen_files),
            'distinct_training_windows_seen':len(seen_windows),
            'available_training_files':len(docs['train']),
            'available_training_windows':len(order),
            'best_checkpoint_step':best_step,
            'best_checkpoint_distinct_training_windows_seen':best_windows,
            'best_checkpoint_distinct_training_files_seen':best_files,
            'batch_size':batch_size,'block_size':block_size,
            'tokenizer_backend':backend,'tokenizer_sha256':file_sha(tokenizer_path),
            'model_parameters':cfg.parameter_count(),
            'first_validation_sample_loss':initial_val,
            'best_validation_sample_loss':best_val,
            'best_full_validation_loss':full_val,
            'heldout_test_loss_once_after_selection':test_loss,
            'test_used_for_model_selection':False,
            'train_last_batch_loss':last_loss,'training_seconds':round(time.monotonic()-start,2),
            'termination':termination,'history':history,
            'best_checkpoint_sha256':file_sha(best_path),
            'last_checkpoint_sha256':file_sha(output/'java-last.pt'),
            'dataset_manifest_sha256':file_sha(corpus/'manifest.json'),
            'java_compile_test':'NOT_RUN',
            'java_instruction_following':'NOT_ESTABLISHED',
            'java_behavior_test':'NOT_RUN',
            'source_legal_review':'PINNED_MIT_APACHE2_LICENSE_TEXT_NOT_LEGAL_ADVICE'}
    (output/'train-report.json').write_text(json.dumps(report,ensure_ascii=False,indent=2)+'\n',encoding='utf8')
    return report


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--corpus',type=Path,required=True)
    p.add_argument('--output',type=Path,required=True)
    p.add_argument('--steps',type=int,default=950)
    p.add_argument('--max-seconds',type=float,default=480.0)
    p.add_argument('--batch-size',type=int,default=6)
    p.add_argument('--block-size',type=int,default=192)
    p.add_argument('--per-file',type=int,default=4)
    p.add_argument('--vocab-size',type=int,default=2048)
    p.add_argument('--backend',choices=('hf','legacy'),default='hf')
    p.add_argument('--lr',type=float,default=.0003)
    p.add_argument('--seed',type=int,default=616)
    p.add_argument('--eval-every',type=int,default=200)
    a=p.parse_args()
    report=train(a.corpus,a.output,steps=a.steps,max_seconds=a.max_seconds,
                 batch_size=a.batch_size,block_size=a.block_size,per_file=a.per_file,
                 vocab_size=a.vocab_size,backend=a.backend,lr=a.lr,
                 seed=a.seed,eval_every=a.eval_every)
    print('JAVA_PUBLIC_TRAIN_SUMMARY '+json.dumps({k:v for k,v in report.items() if k!='history'},ensure_ascii=False),flush=True)