"""Audited, bounded Java instruction SFT of the pinned v0.16 Java code LM.

Input instructions/answers are project-authored v0.15 examples and must be
verified against canonical regeneration. This does not run generated Java.
"""
from __future__ import annotations

from collections import defaultdict
from dataclasses import replace
import hashlib
import json
from pathlib import Path
import random
import re
import tempfile
import time

import torch

from .config import ModelConfig
from .java_diverse_corpus import verify_java_diverse_corpus
from .model import SagaTransformer
from .tokenization import load_tokenizer
from .training import checkpoint_load, checkpoint_save, verify_checkpoint_tokenizer
from .java_vibe import _compile

PROMPT_VERSION = 'compact_java_17_sft_v1'
MAX_CONTEXT_PARAMETERS = 3_000_000
MAX_BASE_CHECKPOINT_BYTES = 90 * 1024 * 1024
# Provenance pinned to the real-source-trained, successfully tested v0.16 artifact.
V016_BASE_BEST_SHA256 = '96005da938d3317fe2ba5185b71cdbb6ed2fee6bc4b9c39e370006d5f2917d1b'
V016_TOKENIZER_SHA256 = '186566808cb03e328cac9a6b7b384a8d0aefc54ccb83fbe97782a751726682fe'
# SHA-256 from the completed v0.17 GitHub Actions SFT experiment (best step 100).
V017_SFT_BEST_SHA256 = 'a17268d80ff443e4a0347f3a29df433556506f13537c78ce282836fa8fe4e5f9'
REQUEST_PATTERN = re.compile(r'User request \(data\):\n([^\n]+)\nCurrent source \(data\):\n', re.M)


def sha_file(path: str | Path) -> str:
    h = hashlib.sha256()
    with Path(path).open('rb') as fh:
        for block in iter(lambda: fh.read(1024*1024), b''):
            h.update(block)
    return h.hexdigest()


def compact_prompt(task: str) -> str:
    """Short enough to leave room for a complete one-class Java answer."""
    if not isinstance(task, str) or not task or len(task) > 240 or '\n' in task or '\x00' in task:
        raise ValueError('Task must be one short line')
    return 'User: Write a complete Java 17 public class Main. Task: ' + task + '\nAssistant:\n'


def source_task(row: dict) -> tuple[str, str]:
    try:
        user, assistant = row['messages']
        if (user['role'],assistant['role']) != ('user','assistant'):
            raise ValueError('Invalid roles')
        m = REQUEST_PATTERN.search(user['content'])
        if not m:
            raise ValueError('Missing original user request')
        task = m.group(1)
        source = assistant['content']
        if 'public class Main' not in source or len(source) > 1800:
            raise ValueError('Expected self-contained Main.java source')
        return task,source
    except (KeyError,TypeError,ValueError) as exc:
        raise ValueError('Invalid Java instruction record') from exc


def encode_row(row: dict,tokenizer,context:int):
    task, source = source_task(row)
    prefix = [tokenizer.SPECIAL['<bos>']] + tokenizer.encode(compact_prompt(task))
    response = tokenizer.encode(source) + [tokenizer.SPECIAL['<eos>']]
    if len(prefix) + len(response)-1 > context:
        return None
    if not response or len(response)>context:
        return None
    ids = prefix+response
    x=torch.tensor(ids[:-1],dtype=torch.long)
    y=torch.tensor(ids[1:],dtype=torch.long)
    y[:len(prefix)-1]=-100
    return {'id':row['id'],'family':row['family'],'task':task,'source':source,
            'x':x,'y':y,'prefix_ids':prefix,'response_tokens':len(response),
            'sequence_positions':len(ids)-1}


def read_split(root:Path, split:str, tokenizer, context:int, *, original_context:int|None=None) -> tuple[list[dict],dict]:
    if split not in ('train','validation','test'):
        raise ValueError('Unknown split')
    items=[];seen=set();families=set();dropped=0;total=0;newly_admitted=0
    for line in (root/(split+'.jsonl')).read_text(encoding='utf8').splitlines():
        row=json.loads(line);total+=1
        if row['id'] in seen:raise ValueError('Repeated row ID')
        seen.add(row['id'])
        encoded=encode_row(row,tokenizer,context)
        if encoded is None:
            dropped+=1
        else:
            items.append(encoded)
            families.add(row['family'])
            if original_context is not None and encoded['sequence_positions']>original_context:
                newly_admitted+=1
    return items,{'total':total,'accepted':len(items),'excluded_too_long':dropped,
                  'families_accepted':len(families),
                  'newly_admitted_beyond_original_context':newly_admitted}


def interleaved_order(items:list[dict],seed:int) -> list[int]:
    """One example per family per cycle, to reduce repeated-template starvation."""
    rng=random.Random(seed)
    buckets=defaultdict(list)
    for idx,item in enumerate(items):buckets[item['family']].append(idx)
    for indices in buckets.values():rng.shuffle(indices)
    names=sorted(buckets)
    rng.shuffle(names)
    result=[]
    while any(buckets.values()):
        for name in names:
            if buckets[name]:result.append(buckets[name].pop())
    return result


def eval_sample(items:list[dict], count:int, seed:int=47)->list[dict]:
    order=interleaved_order(items,seed)
    return [items[i] for i in order[:count]]


def pack(items:list[dict]) -> tuple[torch.Tensor,torch.Tensor]:
    if not items:raise ValueError('Empty minibatch')
    length=max(len(item['x']) for item in items)
    x=torch.zeros(len(items),length,dtype=torch.long)
    y=torch.full((len(items),length),-100,dtype=torch.long)
    for i,item in enumerate(items):
        n=len(item['x']);x[i,:n]=item['x'];y[i,:n]=item['y']
    return x,y


def response_loss(model:SagaTransformer,items:list[dict], batch_size:int=6)->float:
    if not items:raise ValueError('No scored examples')
    before=model.training;model.eval();loss_sum=0.;n=0
    with torch.inference_mode():
        for i in range(0,len(items),batch_size):
            x,y=pack(items[i:i+batch_size]);tokens=int((y!=-100).sum())
            loss=model(x,labels=y).loss
            loss_sum+=float(loss)*tokens;n+=tokens
    model.train(before)
    return loss_sum/n


def _validate_base(checkpoint:Path,tokenizer_file:Path,*,require_pinned:bool,
                   pinned_checkpoint_sha256:str=V016_BASE_BEST_SHA256):
    if (not checkpoint.is_file() or checkpoint.is_symlink() or
        checkpoint.stat().st_size>MAX_BASE_CHECKPOINT_BYTES or
        not tokenizer_file.is_file() or tokenizer_file.is_symlink()):
        raise ValueError('Missing, symlinked or oversized model files')
    checkpoint_sha=sha_file(checkpoint);tok_sha=sha_file(tokenizer_file)
    if require_pinned and (checkpoint_sha!=pinned_checkpoint_sha256 or tok_sha!=V016_TOKENIZER_SHA256):
        raise ValueError('The source checkpoint/tokenizer differs from pinned real-Java v0.16 training')
    state=torch.load(checkpoint,map_location='cpu',weights_only=True)
    if not state.get('train_state',{}).get('tokenizer_sha256'):
        raise ValueError('Checkpoint is missing a tokenizer identity hash')
    verify_checkpoint_tokenizer(state,tokenizer_file)
    cfg=ModelConfig(**state['config'])
    if cfg.parameter_count()>MAX_CONTEXT_PARAMETERS or cfg.max_position_embeddings>512:
        raise ValueError('Model outside small-CPU safety budget')
    return state,cfg,checkpoint_sha,tok_sha


def extend_rope_context(config:ModelConfig, target_context:int|None) -> ModelConfig:
    """Change the RoPE position *limit*, not the learned tensor shapes.

    This is extrapolation and cannot establish quality at new positions. A
    maximum of 512 and 2x the original length keeps the CPU experiment bounded.
    """
    if target_context is None:
        return config
    if not isinstance(target_context,int) or isinstance(target_context,bool):
        raise ValueError('Target context must be an integer')
    if not config.max_position_embeddings < target_context <= min(512,2*config.max_position_embeddings):
        raise ValueError('Only bounded RoPE context extension (<=2x and <=512) is allowed')
    extended=replace(config,max_position_embeddings=target_context)
    if extended.parameter_count()!=config.parameter_count():
        raise ValueError('Context extension must not allocate new parameters')
    return extended


def restore_verified_base_weights(model:SagaTransformer, checkpoint_state:dict,
                                  original_config:ModelConfig) -> None:
    """Trust a SHA-checked local checkpoint, verify all shapes and load strictly.

    `checkpoint_load` appropriately rejects config mismatches; this narrowly
    permits a *RoPE limit only* extension with exactly the same learned weights.
    Optimizer state is intentionally discarded for a new training stage.
    """
    if checkpoint_state.get('config')!=original_config.to_dict():
        raise ValueError('Original checkpoint configuration differs from expected')
    expected=extend_rope_context(original_config,model.config.max_position_embeddings) if (model.config.max_position_embeddings!=original_config.max_position_embeddings) else original_config
    if model.config!=expected:
        raise ValueError('Only the RoPE context bound may change')
    model.load_state_dict(checkpoint_state['model'],strict=True)


def classify_javac_diagnostic(stderr: str) -> str:
    """Classify compiler errors using a fixed allowlist; never retain raw stderr.

    This is a coarse diagnostic hint, not a verified root-cause attribution.
    """
    if not isinstance(stderr, str) or not stderr:
        return 'NONE'
    # The first error is the primary compiler diagnostic. Ignore source snippets
    # and filenames; no untrusted compiler text is copied to the report.
    lines = stderr[:16384].splitlines()
    first = next((line.split('error:', 1)[1].strip().lower()
                  for line in lines if 'error:' in line), '')
    if not first:
        return 'OTHER'
    if any(term in first for term in (
            'cannot find symbol', 'package does not exist', 'cannot resolve symbol')):
        return 'MISSING_SYMBOL'
    if any(term in first for term in (
            'incompatible types', 'cannot be converted', 'bad operand types',
            'no suitable method found', 'method cannot be applied')):
        return 'TYPE_ERROR'
    if 'missing return statement' in first:
        return 'MISSING_RETURN'
    if any(term in first for term in (
            "';' expected", "'}' expected", "')' expected", "'(' expected",
            "illegal start of", "reached end of file while parsing",
            "not a statement", "class, interface, enum, or record expected",
            "unclosed string literal", "unclosed character literal")):
        return 'SYNTAX_ERROR'
    return 'OTHER'


def heldout_compilation(model, tokenizer, rows:list[dict], *, max_cases:int=10) -> dict:
    """Compiles model output, never executes it. Called only after model selection."""
    cases=eval_sample(rows,max_cases,seed=67)
    if not cases:
        return {'status':'NOT_RUN','reason':'No accepted test examples'}
    total=0;compiles=0;exact=0;details=[]
    for row in cases:
        total+=1
        prefix=row['prefix_ids']
        remaining=model.config.max_position_embeddings-len(prefix)
        if remaining<2:
            details.append({'id':row['id'],'status':'INSUFFICIENT_CONTEXT'});continue
        with torch.inference_mode():
            sequence=model.generate(torch.tensor([prefix],dtype=torch.long),
                max_new_tokens=min(remaining,170),temperature=0.0,
                eos_token_id=tokenizer.SPECIAL['<eos>'],vocab_limit=tokenizer.vocab_size,
                deadline_monotonic=time.monotonic()+25)
        generated_ids=sequence[0,len(prefix):].tolist()
        output=tokenizer.decode(generated_ids)
        identical=output.strip()==row['source'].strip()
        exact+=int(identical)
        status='FAIL'
        reason='NOT_COMPILED'
        diagnostic_hash=None
        diagnostic_category='NONE'
        javac_error_category='NONE'
        if len(output)>=4000:
            reason='OUTPUT_TOO_LONG'
        elif 'public class Main' not in output:
            reason='MISSING_PUBLIC_MAIN'
        elif '\x00' in output:
            reason='NUL_BYTE'
        else:
            with tempfile.TemporaryDirectory(prefix='sagallm-sft-javac-') as location:
                root=Path(location);source=root/'Main.java';source.write_text(output,encoding='utf8')
                result=_compile(source,root,timeout=12)
                status=result['status']
                reason=result.get('reason','JAVAC')
                stderr=result.get('diagnostics','')
                if stderr:
                    diagnostic_hash=hashlib.sha256(stderr.encode('utf8')).hexdigest()
                    diagnostic_category='JAVAC_DIAGNOSTIC'
                    if status=='FAIL':
                        javac_error_category=classify_javac_diagnostic(stderr)
                elif status=='FAIL':
                    diagnostic_category='JAVAC_FAILURE_NO_STDERR'
        compiles+=int(status=='PASS')
        eos=tokenizer.SPECIAL['<eos>']
        details.append({'id':row['id'],'family':row['family'],
                        'compile':status,'exact':identical,
                        'generated_sha256':hashlib.sha256(output.encode('utf8')).hexdigest(),
                        'generated_tokens':len(generated_ids),
                        'generation_eos_seen':eos in generated_ids,
                        'generation_hit_token_cap':len(generated_ids)>=min(remaining,170) and eos not in generated_ids,
                        'generation_stop_reason':'EOS' if eos in generated_ids else
                            'TOKEN_CAP_OR_DEADLINE_UNKNOWN',
                        'compile_reason':reason,
                        'javac_stderr_sha256':diagnostic_hash,
                        'javac_diagnostic_category':diagnostic_category,
                        'javac_error_category':javac_error_category,
                        'source_persisted':False,
                        'java_behavior':'NOT_RUN'})
    return {'status':'SCORED_COMPILE_ONLY','evaluated':total,'compiled':compiles,
            'exact_reference_matches':exact,'behavior':'NOT_RUN','cases':details}


def train_instruction_sft(base_checkpoint:str|Path,tokenizer_file:str|Path,
                          dataset:str|Path,output:str|Path,*,steps:int=500,
                          batch_size:int=4,eval_every:int=100,
                          eval_count:int=30,test_count:int=30,compile_cases:int=10,
                          lr:float=0.00015,seed:int=717,max_seconds:float=360,
                          require_pinned:bool=True, target_context:int|None=None,
                          pinned_checkpoint_sha256:str=V016_BASE_BEST_SHA256)->dict:
    if not (1<=steps<=1500 and 1<=batch_size<=8 and 1<=eval_every<=500
            and 1<=eval_count<=100 and 1<=test_count<=100 and 0<=compile_cases<=12
            and 0<lr<=0.001 and 10<=max_seconds<=600):
        raise ValueError('Unsafe SFT budget')
    root=Path(dataset);dest=Path(output)
    if dest.exists() or dest.is_symlink():raise FileExistsError('Refuse to replace experiment output')
    # Verify ALL splits before allocating a model. This recomputes canonical data.
    verify_java_diverse_corpus(root)
    state,original_cfg,base_sha,tok_sha=_validate_base(
        Path(base_checkpoint),Path(tokenizer_file),require_pinned=require_pinned,
        pinned_checkpoint_sha256=pinned_checkpoint_sha256)
    cfg=extend_rope_context(original_cfg,target_context)
    tokenizer=load_tokenizer(tokenizer_file)
    if tokenizer.vocab_size>cfg.vocab_size:raise ValueError('Checkpoint vocabulary mismatch')
    train_rows,train_info=read_split(root,'train',tokenizer,cfg.max_position_embeddings,
        original_context=original_cfg.max_position_embeddings)
    valid_rows,valid_info=read_split(root,'validation',tokenizer,cfg.max_position_embeddings,
        original_context=original_cfg.max_position_embeddings)
    # Do not decode, tokenize, inspect or score TEST until best checkpoint chosen.
    if len(train_rows)<batch_size or not valid_rows:
        raise ValueError('Too few complete SFT examples for this context size')
    if len({r['family'] for r in train_rows})<10:
        raise ValueError('Insufficient task-family diversity after context filtering')
    if len({r['family'] for r in valid_rows})<2:
        raise ValueError('Insufficient validation-family diversity')
    torch.set_num_threads(2)
    torch.manual_seed(seed)
    model=SagaTransformer(cfg)
    restore_verified_base_weights(model,state,original_cfg)
    optimizer=torch.optim.AdamW(model.parameters(),lr=lr,weight_decay=0.01)
    validation=eval_sample(valid_rows,eval_count)
    baseline=response_loss(model,validation)
    dest.mkdir(parents=True)
    tok_target=dest/'tokenizer.json';tok_target.write_bytes(Path(tokenizer_file).read_bytes())
    metadata={'stage':'java_instruction_sft_v018' if target_context else 'java_instruction_sft_v017',
              'original_context_length':original_cfg.max_position_embeddings,
              'trained_context_length':cfg.max_position_embeddings,
              'context_extension':'ROPE_EXTRAPOLATION_UNVALIDATED' if target_context else 'NONE','tokenizer_sha256':tok_sha,
              'base_checkpoint_sha256':base_sha,'dataset_manifest_sha256':sha_file(root/'manifest.json'),
              'prompt_version':PROMPT_VERSION,'optimizer_base_lr':lr,'seed':seed}
    best_path=dest/'java-sft-best.pt'
    checkpoint_save(best_path,model,optimizer,0,cfg,{**metadata,'selected_validation_response_loss':baseline})
    order=interleaved_order(train_rows,seed);seen=set()
    train_start=time.monotonic();history=[];best_loss=baseline;best_step=0
    best_seen=0;completed=0;last_batch_loss=None;termination='STEP_LIMIT'
    for step in range(1,steps+1):
        if time.monotonic()-train_start>max_seconds:
            termination='TIME_BUDGET';break
        current=[]
        for j in range(batch_size):
            pos=((step-1)*batch_size+j)%len(order)
            index=order[pos];current.append(train_rows[index]);seen.add(index)
        x,y=pack(current)
        optimizer.zero_grad(set_to_none=True)
        loss=model(x,labels=y).loss
        if not bool(torch.isfinite(loss)):raise FloatingPointError('Nonfinite SFT loss')
        loss.backward();torch.nn.utils.clip_grad_norm_(model.parameters(),1.0);optimizer.step()
        last_batch_loss=float(loss.detach());completed=step
        if step%eval_every==0 or step==steps:
            val=response_loss(model,validation)
            record={'step':step,'last_train_response_loss':last_batch_loss,
                    'validation_response_loss':val,'distinct_train_examples_seen':len(seen),
                    'distinct_train_families_seen':len({train_rows[i]['family'] for i in seen})}
            history.append(record);print(json.dumps(record),flush=True)
            if val<best_loss-0.001:
                best_loss=val;best_step=step;best_seen=len(seen)
                checkpoint_save(best_path,model,optimizer,step,cfg,
                                {**metadata,'selected_validation_response_loss':val,
                                 'distinct_training_examples_seen':best_seen})
    checkpoint_save(dest/'java-sft-last.pt',model,optimizer,completed,cfg,
                    {**metadata,'selection':'last','distinct_training_examples_seen':len(seen)})
    # TEST first becomes accessible after best checkpoint selection is complete.
    test_rows,test_info=read_split(root,'test',tokenizer,cfg.max_position_embeddings,
        original_context=original_cfg.max_position_embeddings)
    if not test_rows:raise ValueError('No accepted held-out test examples')
    test_sample=eval_sample(test_rows,test_count)
    last_test=response_loss(model,test_sample)
    checkpoint_load(best_path,model)
    best_test=response_loss(model,test_sample)
    # Baseline diagnostic on the same held-out tests, after all checkpoint selection.
    restore_verified_base_weights(model,state,original_cfg)
    base_test=response_loss(model,test_sample)
    # Compile-only generation: baseline and best on the exact same heldout prompts.
    baseline_compile={'status':'NOT_RUN'};best_compile={'status':'NOT_RUN'}
    if compile_cases:
        baseline_compile=heldout_compilation(model,tokenizer,test_rows,max_cases=compile_cases)
        checkpoint_load(best_path,model)
        best_compile=heldout_compilation(model,tokenizer,test_rows,max_cases=compile_cases)
    report={'status':'TRAINED_CPU','phase':'JAVA_INSTRUCTION_SFT','model_origin':'PINNED_REAL_JAVA_V017_SFT' if target_context else 'PINNED_REAL_JAVA_V016',
            'checkpoint_sha256_base':base_sha,'tokenizer_sha256':tok_sha,
            'data_manifest_sha256':sha_file(root/'manifest.json'),'prompt_version':PROMPT_VERSION,
            'dataset_source':'project-authored parametric synthetic Java v0.15; NOT independent human tasks',
            'original_split_counts':{'train':train_info,'validation':valid_info,'test':test_info},
            'model_parameters':cfg.parameter_count(),'model_context_length':cfg.max_position_embeddings,
            'original_model_context_length':original_cfg.max_position_embeddings,
            'context_extension':'ROPE_EXTRAPOLATION_UNVALIDATED' if target_context else 'NONE',
            'optimizer_steps_requested':steps,'optimizer_steps_completed':completed,'termination':termination,
            'distinct_training_examples_seen':len(seen),
            'best_checkpoint_step':best_step,'best_checkpoint_train_examples_seen':best_seen,
            'initial_validation_response_loss':baseline,'best_validation_response_loss':best_loss,
            'heldout_test_response_loss_before_sft':base_test,
            'heldout_test_response_loss_best_sft':best_test,
            'heldout_test_response_loss_last_sft':last_test,
            'test_used_for_model_selection':False,
            'compile_only_base':baseline_compile,'compile_only_best':best_compile,
            'java_code_executed':False,'java_behavior_test':'NOT_RUN',
            'java_coding_ability':'NOT_ESTABLISHED','history':history,
            'best_checkpoint_sha256':sha_file(best_path),
            'last_checkpoint_sha256':sha_file(dest/'java-sft-last.pt'),
            'training_seconds':round(time.monotonic()-train_start,2)}
    (dest/'train-report.json').write_text(json.dumps(report,ensure_ascii=False,indent=2)+'\n',encoding='utf8')
    return report