"""One-pass, train-family-only original Java SFT; CPU-first and reproducible.

No pretrained weights, downloaded code, external services, or executable model
output. Report both corpus size and distinct examples that received gradients.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import random
import shutil
import subprocess
import tempfile
import time

import torch

from sagallm.config import ModelConfig
from sagallm.java_large_corpus import verify_java_large_corpus, TEST_FAMILIES, TRAIN_FAMILIES
from sagallm.java_ai import LocalCheckpointGenerator, extract_java
from sagallm.model import SagaTransformer
from sagallm.post_training import format_sft_example
from sagallm.tokenization import ByteBPETokenizer, load_tokenizer
from sagallm.training import checkpoint_load, checkpoint_save, verify_checkpoint_tokenizer


def digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def rows(path: Path):
    return [json.loads(line) for line in path.read_text(encoding='utf-8').splitlines() if line]


def loss_mean(model, pairs):
    old=model.training;model.eval();s=0.
    with torch.inference_mode():
        for x,y in pairs:
            s+=float(model(x.unsqueeze(0),labels=y.unsqueeze(0)).loss)
    model.train(old)
    return s/len(pairs)


def evaluate_compilation(dataset: Path, checkpoint: Path, tokenizer: Path, limit: int = 12) -> dict:
    """Compile-only external heldout family examples; never execute Java."""
    if shutil.which('javac') is None:
        return {'status':'NOT_RUN','reason':'JDK unavailable','evaluated':0}
    generator=LocalCheckpointGenerator(checkpoint,tokenizer,device='cpu')
    cases=rows(dataset/'test.jsonl')
    selected=[]
    for i in range(18):
        for f in TEST_FAMILIES:
            candidates=[item for item in cases if item['family']==f]
            if i < len(candidates): selected.append(candidates[i])
    records=[]
    for item in selected[:limit]:
        prompt='User: '+item['messages'][0]['content']+'\nAssistant: '
        try:
            text=generator(prompt,max_new_tokens=170)
            code=extract_java(text,'Main')
            exact=code.strip()==item['messages'][1]['content'].strip()
            with tempfile.TemporaryDirectory() as tmp:
                f=Path(tmp)/'Main.java';f.write_text(code,encoding='utf-8')
                result=subprocess.run(['javac','-proc:none','-d',tmp,str(f)],cwd=tmp,
                                      capture_output=True,text=True,timeout=12,check=False)
                compiled=result.returncode==0
        except (ValueError,TypeError,subprocess.TimeoutExpired,TimeoutError):
            compiled=exact=False
        records.append({'id':item['id'],'family':item['family'],'compiled':compiled,'exact_reference_match':exact})
    return {'status':'PASS_RUNTIME','evaluated':len(records),
            'compiled':sum(x['compiled'] for x in records),
            'exact_reference_matches':sum(x['exact_reference_match'] for x in records),
            'behavioral_execution':'NOT_RUN','semantic_correctness':'NOT_ESTABLISHED','items':records}


def train(dataset: Path, output: Path, *, steps: int = 5220, max_seconds: float = 700.,
          hidden_size: int = 160, layers: int = 3, lr: float = 0.0008,
          progress_every: int = 300, seed: int = 20261122, compile_limit: int = 12,
          init_checkpoint: Path | None = None, init_tokenizer: Path | None = None) -> dict:
    if not 1<=steps<=15000 or not 15<=max_seconds<=1800 or not 1<=progress_every<=2000 or not 0<lr<0.1:
        raise ValueError('Unsupported optimizer budget')
    if hidden_size not in (128,160,192) or layers not in (2,3,4) or not 0<=compile_limit<=18:
        raise ValueError('Unsupported model configuration/evaluation count')
    if output.exists():raise FileExistsError('Never overwrite an experiment')
    if (init_checkpoint is None) != (init_tokenizer is None):
        raise ValueError('init_checkpoint and init_tokenizer must be specified together')
    if init_checkpoint is not None and (init_checkpoint.stat().st_size>128_000_000 or init_checkpoint.is_symlink()):
        raise ValueError('Refusing oversized or symlinked previous checkpoint')
    audit=verify_java_large_corpus(dataset)
    data={split:rows(dataset/(split+'.jsonl')) for split in ('train','validation')}
    torch.set_num_threads(min(2,max(1,os.cpu_count() or 1)))
    torch.manual_seed(seed)
    random.seed(seed)
    groups={f:[r for r in data['train'] if r['family']==f] for f in TRAIN_FAMILIES}
    # BPE is fitted to training texts only, with family-balanced sampling.
    tokenizer_sample=[r for f in TRAIN_FAMILIES for r in groups[f][:2]]
    if init_checkpoint is not None:
        previous_state=torch.load(init_checkpoint,map_location='cpu',weights_only=True)
        verify_checkpoint_tokenizer(previous_state,init_tokenizer)
        cfg=ModelConfig(**previous_state['config'])
        tokenizer=load_tokenizer(init_tokenizer)
        if tokenizer.vocab_size>cfg.vocab_size:
            raise ValueError('Existing tokenizer exceeds prior model vocabulary')
    else:
        tokenizer=ByteBPETokenizer.train((m['content'] for r in tokenizer_sample for m in r['messages']),
                                         vocab_size=512,min_frequency=2)
        cfg=ModelConfig(vocab_size=512,hidden_size=hidden_size,num_layers=layers,
                        num_heads=4,num_kv_heads=2,intermediate_size=hidden_size*2,
                        max_position_embeddings=384)
    if cfg.parameter_count()>2_000_000:raise ValueError('Exceeds experimental parameter guard')
    pairs={split:[format_sft_example(r['messages'],tokenizer,cfg.max_position_embeddings)
                  for r in entries] for split,entries in data.items()}
    # Fixed validation subsample for selection. Full validation only measured
    # AFTER selecting the checkpoint; TEST is not even read until that point.
    val_sample=[]
    for f in sorted({r['family'] for r in data['validation']}):
        val_sample.extend(pairs['validation'][i] for i,row in enumerate(data['validation']) if row['family']==f)  # 18 per family
    validation_selection=val_sample[::9]
    if not validation_selection:raise ValueError('No validation rows')
    output.mkdir(parents=True)
    tokenizer_path=output/'tokenizer.json';tokenizer.save(tokenizer_path)
    model=SagaTransformer(cfg).train()
    if init_checkpoint is not None:
        checkpoint_load(init_checkpoint,model)  # weights only; new dataset gets a fresh optimizer
    optimizer=torch.optim.AdamW(model.parameters(),lr=lr,weight_decay=0.01)
    meta={'stage':'original_java_expanded_random_init','tokenizer_sha256':digest(tokenizer_path),
          'training_dataset_sha256':digest(dataset/'train.jsonl'),
          'validation_dataset_sha256':digest(dataset/'validation.jsonl'),
          'test_dataset_sha256':digest(dataset/'test.jsonl'),
          'dataset_manifest_sha256':digest(dataset/'manifest.json'),
          'seed':seed,'lr':lr,'version':'expanded-v1',
          'previous_local_checkpoint_sha256':digest(init_checkpoint) if init_checkpoint else None}
    initial=loss_mean(model,pairs['train'][:16]);v0=loss_mean(model,validation_selection)
    best=output/'java-best.pt'
    checkpoint_save(best,model,optimizer,0,cfg,{**meta,'best_validation_loss':v0,'selection_rule':'validation_only'})
    order=list(range(len(pairs['train'])));random.Random(seed).shuffle(order)
    seen=set();best_val=v0;best_step=0;history=[{'step':0,'train_sample_loss':initial,'val_sample_loss':v0}]
    start=time.monotonic();completed=0;stop='max_steps'
    for step in range(1,steps+1):
        if time.monotonic()-start>max_seconds:
            stop='time_budget';break
        # Visit every training example before resampling a single one.
        row_index=order[(step-1)%len(order)];seen.add(row_index)
        x,y=pairs['train'][row_index]
        optimizer.zero_grad(set_to_none=True)
        loss=model(x.unsqueeze(0),labels=y.unsqueeze(0)).loss
        if not torch.isfinite(loss):raise FloatingPointError('Non-finite Java loss')
        loss.backward();torch.nn.utils.clip_grad_norm_(model.parameters(),1.)
        optimizer.step();completed=step
        if step%progress_every==0 or step==steps:
            vl=loss_mean(model,validation_selection)
            tr=loss_mean(model,pairs['train'][:16])
            history.append({'step':step,'train_sample_loss':tr,'val_sample_loss':vl})
            print(f'step={step} distinct_train={len(seen)}/{len(order)} train_sample_loss={tr:.4f} val_sample_loss={vl:.4f} elapsed={time.monotonic()-start:.1f}s',flush=True)
            if vl<best_val-1e-3:
                best_val=vl;best_step=step
                checkpoint_save(best,model,optimizer,step,cfg,{**meta,'best_validation_loss':vl,'selection_rule':'validation_only'})
    last=output/'java-last.pt'
    checkpoint_save(last,model,optimizer,completed,cfg,{**meta,'selection_rule':'last_not_selected'})
    checkpoint_load(best,model)
    model.eval()
    full_val=loss_mean(model,pairs['validation'])
    # The integrity verifier has checked TEST bytes already, but its targets
    # are only tokenized and scored after validation-only model selection.
    test=rows(dataset/'test.jsonl')
    test_pairs=[format_sft_example(item['messages'],tokenizer,cfg.max_position_embeddings) for item in test]
    test_loss=loss_mean(model,test_pairs)
    del model,optimizer
    compile_report=(evaluate_compilation(dataset,best,tokenizer_path,compile_limit)
                    if compile_limit else {'status':'NOT_RUN','reason':'compile_limit=0'})
    report={
        'status':'TRAINED_CPU',
        'model_weights_origin':'PRIOR_LOCAL_SAGALLM' if init_checkpoint else 'RANDOM_INIT',
        'external_pretrained_model_used':False,
        'initial_local_checkpoint_sha256':digest(init_checkpoint) if init_checkpoint else None,
        'corpus_counts':audit['counts'],'total_corpus_rows':sum(audit['counts'].values()),
        'optimizer_steps_requested':steps,'optimizer_steps_completed':completed,
        'distinct_training_examples_seen':len(seen),'total_training_examples':len(order),
        'best_checkpoint_distinct_training_examples_seen':min(best_step,len(order)),
        'unseen_training_examples':len(order)-len(seen),
        'training_seconds':round(time.monotonic()-start,2),'termination':stop,
        'parameter_count':cfg.parameter_count(),'context_length':cfg.max_position_embeddings,
        'tokenizer_vocab_size':tokenizer.vocab_size,'seed':seed,
        'train_sample_loss_before':initial,'train_sample_loss_after':history[-1]['train_sample_loss'],
        'validation_selection_loss_before':v0,'validation_selection_loss_best':best_val,
        'validation_full_loss_best_checkpoint':full_val,'test_loss_selected_checkpoint':test_loss,
        'test_used_for_model_selection':False,'test_loss_evaluation_count':1,
        'best_checkpoint_step':best_step,'history':history,'compile_only_heldout':compile_report,
        'checkpoint_sha256':digest(best),'last_checkpoint_sha256':digest(last),
        'tokenizer_sha256':digest(tokenizer_path),'corpus_manifest_sha256':digest(dataset/'manifest.json'),
        'model_config':cfg.to_dict(),'semantic_java_ability':'NOT_ESTABLISHED',
        'model_generated_java_behavior':'NOT_RUN',
        'legal_status':'ORIGINAL_ASSISTANT_AUTHORED_NOT_LEGALLY_ATTESTED',
        'known_limitation':'Many parameterized examples of 70 templates do not equal 5436 distinct algorithms.',
    }
    (output/'train-report.json').write_text(json.dumps(report,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
    return report

if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--dataset',type=Path,required=True)
    p.add_argument('--output',type=Path,required=True)
    p.add_argument('--steps',type=int,default=5220)
    p.add_argument('--max-seconds',type=float,default=700.)
    p.add_argument('--hidden-size',type=int,default=160)
    p.add_argument('--layers',type=int,default=3)
    p.add_argument('--lr',type=float,default=.0008)
    p.add_argument('--progress-every',type=int,default=300)
    p.add_argument('--seed',type=int,default=20261122)
    p.add_argument('--compile-limit',type=int,default=12)
    p.add_argument('--init-checkpoint',type=Path)
    p.add_argument('--init-tokenizer',type=Path)
    a=p.parse_args()
    result=train(a.dataset,a.output,steps=a.steps,max_seconds=a.max_seconds,hidden_size=a.hidden_size,
                 layers=a.layers,lr=a.lr,progress_every=a.progress_every,seed=a.seed,
                 compile_limit=a.compile_limit,init_checkpoint=a.init_checkpoint,
                 init_tokenizer=a.init_tokenizer)
    print(json.dumps({k:v for k,v in result.items() if k not in ('model_config','history')},ensure_ascii=False,indent=2))