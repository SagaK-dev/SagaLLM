"""Offline inventory/export of Java source code with explicit unreviewed provenance.

This does NOT confer copyright/license rights or approve training. The caller
must independently review rights and use a SHA-pinned dataset card before use.
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path
import re

from .curation import sensitive_reason, low_quality

MAX_FILES=50000
MAX_JAVA_BYTES=256*1024
MAX_TOTAL_BYTES=256*1024*1024
EXCLUDED=frozenset({'.git','.gradle','.idea','target','build','out','node_modules','vendor','generated','dist'})


def _hash_bytes(raw: bytes) -> str:
    return hashlib.sha256(raw).hexdigest()


def inventory_java_tree(root: str|Path) -> dict:
    """Stateless scan: skip symlinks, generated folders, sensitive and duplicate code."""
    folder=Path(root)
    if not folder.is_dir() or folder.is_symlink():
        raise ValueError('Source must be a real local directory, not a symlink')
    files=[];rejected={};unique=set();total=0
    for file in sorted(folder.rglob('*.java')):
        rel=file.relative_to(folder)
        if any(segment in EXCLUDED or segment.startswith('.') for segment in rel.parts[:-1]):
            rejected['excluded_directory']=rejected.get('excluded_directory',0)+1;continue
        if file.is_symlink() or not file.is_file() or file.stat().st_size>MAX_JAVA_BYTES:
            rejected['unsafe_or_large']=rejected.get('unsafe_or_large',0)+1;continue
        if len(files)>=MAX_FILES:raise ValueError('Java file inventory exceeds maximum entries')
        raw=file.read_bytes();total+=len(raw)
        if total>MAX_TOTAL_BYTES:raise ValueError('Java source inventory exceeds 256 MiB')
        if b'\x00' in raw:
            rejected['binary']=rejected.get('binary',0)+1;continue
        try:
            code=raw.decode('utf8')
        except UnicodeError:
            rejected['not_utf8']=rejected.get('not_utf8',0)+1;continue
        if low_quality(code):
            rejected['low_quality']=rejected.get('low_quality',0)+1;continue
        sensitive=sensitive_reason(code)
        if sensitive:
            rejected['sensitive_'+sensitive]=rejected.get('sensitive_'+sensitive,0)+1;continue
        code=code.replace('\r\n','\n').replace('\r','\n')
        fingerprint=_hash_bytes(code.encode('utf8'))
        if fingerprint in unique:
            rejected['duplicate']=rejected.get('duplicate',0)+1;continue
        unique.add(fingerprint)
        files.append({'relative_path':rel.as_posix(),'sha256':fingerprint,'source':code})
    return {'records':files,'rejections':rejected,'bytes_seen':total,'safe_files':len(files)}


def export_java_source_corpus(source: str|Path,output: str|Path,*,
                              source_uri: str, revision: str, license_id: str='UNVERIFIED') -> dict:
    """Export non-approved Java sources for human review; never silently allow training."""
    if not isinstance(source_uri,str) or not source_uri.strip() or len(source_uri)>512:
        raise ValueError('A specific repository source URL is required')
    if not isinstance(revision,str) or not re.fullmatch(r'[0-9a-f]{40}',revision):
        raise ValueError('Expected a pinned 40-character Git revision')
    if license_id!='UNVERIFIED':
        raise ValueError('Do not claim a license automatically; approve separately with a dataset card')
    dest=Path(output).expanduser().absolute()
    if dest.exists():raise FileExistsError('Never overwrite exported corpus')
    inventory=inventory_java_tree(source)
    if not inventory['safe_files']:raise ValueError('No locally readable Java source files')
    dest.mkdir(parents=True)
    try:
        content=('\n'.join(json.dumps({'text':x['source'],
                                         'source_path':x['relative_path'],
                                         'source_sha256':x['sha256']},
                                         ensure_ascii=False,sort_keys=True,separators=(',',':'))
                           for x in inventory['records'])+'\n').encode('utf8')
        raw=dest/'java-documents.jsonl';raw.write_bytes(content)
        manifest={'schema':'sagallm-java-repository-inventory-v1','source_uri':source_uri,
                  'revision_declared_not_verified':revision,'review_status':'PENDING',
                  'license_id':'UNVERIFIED','training_authorized':False,
                  'java_files':inventory['safe_files'],'skipped':inventory['rejections'],
                  'java_documents_sha256':_hash_bytes(content),
                  'warning':'No license/right approval; verify a purpose-specific dataset card before training.'}
        (dest/'manifest.json').write_text(json.dumps(manifest,sort_keys=True,indent=2)+'\n',encoding='utf8')
        return {k:v for k,v in manifest.items() if k not in ('source_uri','revision_declared_not_verified')}
    except BaseException:
        import shutil
        shutil.rmtree(dest)
        raise