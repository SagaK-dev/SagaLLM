"""Offline Java source corpus with pinned, independently checked upstream evidence.

Only the documented commits/license files may be used. This is an engineering
provenance check, not a legal conclusion that every upstream contribution is
lawfully sublicensable or that the training use is lawful everywhere.
"""
from __future__ import annotations

from collections import Counter
from dataclasses import dataclass
import hashlib
import json
from pathlib import Path
import re
import subprocess

from .curation import sensitive_reason, low_quality


@dataclass(frozen=True)
class Upstream:
    slug: str
    url: str
    revision: str
    license_path: str
    license_blob: str
    license_id: str


UPSTREAMS = (
    Upstream('algorithms', 'https://github.com/TheAlgorithms/Java',
             'f4d6708297f86629a9e415070935b903ea8b1858', 'LICENSE',
             'f6bcf04e7773b3bfca7a30836672dc7322e38944', 'MIT'),
    Upstream('gson', 'https://github.com/google/gson',
             'd9b5079eaa4567219771bba6d04289cf45c7589b', 'LICENSE',
             'd645695673349e3947e8e5ae42332d0ac3164cd7', 'Apache-2.0'),
    Upstream('guava', 'https://github.com/google/guava',
             '36e3995499d6c3b74b1207d362fa9e19915d2589', 'LICENSE',
             'd645695673349e3947e8e5ae42332d0ac3164cd7', 'Apache-2.0'),
    Upstream('patterns', 'https://github.com/iluwatar/java-design-patterns',
             '4cabb204f240240bea319605a5ffa46ded728001', 'LICENSE.md',
             'bcae7daadee982a10f5f9497f493eb4d70cd517f', 'MIT'),
)
FORMAT = 'sagallm-java-pinned-open-source-v1'
MAX_BYTES_PER_FILE = 96_000
MAX_TOTAL_BYTES = 96_000_000
MAX_FILES = 12_000


def sha256(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def git_blob_sha(data: bytes) -> str:
    return hashlib.sha1(b'blob ' + str(len(data)).encode('ascii') + b'\0' + data).hexdigest()


def _git(root: Path, *args: str) -> str:
    r = subprocess.run(['git', '-C', str(root), *args], capture_output=True,
                       text=True, timeout=12, check=False)
    if r.returncode:
        raise ValueError('Git provenance could not be verified')
    return r.stdout.strip()


def verify_upstream(root: Path, spec: Upstream) -> bytes:
    if not root.is_dir() or root.is_symlink():
        raise ValueError('Upstream root must be a non-symlink directory')
    if _git(root, 'rev-parse', 'HEAD') != spec.revision:
        raise ValueError('Git revision differs from the pinned source')
    if _git(root, 'status', '--porcelain', '--untracked-files=no'):
        raise ValueError('Tracked upstream files have uncommitted changes')
    if _git(root, 'hash-object', '--', spec.license_path) != spec.license_blob:
        raise ValueError('Git license blob differs from reviewed source')
    license_file = root / spec.license_path
    if license_file.is_symlink():
        raise ValueError('License cannot be a symlink')
    raw = license_file.read_bytes()
    if git_blob_sha(raw) != spec.license_blob:
        raise ValueError('License contents do not match pinned Git blob')
    start = raw.decode('utf-8', errors='replace')[:300].lower()
    if spec.license_id == 'MIT' and 'mit license' not in start:
        raise ValueError('Unexpected MIT license format')
    if spec.license_id == 'Apache-2.0' and 'apache license' not in start:
        raise ValueError('Unexpected Apache license format')
    return raw


def _is_source(path: str, slug: str) -> bool:
    """Production Java only: no test fixtures, Android copies, generated or vendor paths."""
    if not path.endswith('.java'):
        return False
    parts = path.split('/')
    if any(x in {'test','tests','generated','build','target','out','vendor',
                  'resources','src-super','examples'} for x in parts):
        return False
    if slug == 'algorithms':
        return path.startswith('src/main/java/')
    if slug == 'guava':
        return path.startswith('guava/src/com/google/common/')
    return '/src/main/java/' in '/' + path


def category_key(slug: str, path: str) -> str:
    """Disjoint source-package groups limit train/validation/test related-file leakage."""
    if slug == 'patterns':
        return f'{slug}/{path.split("/", 1)[0]}'
    if slug == 'algorithms':
        rest = path.removeprefix('src/main/java/com/thealgorithms/')
        return f'{slug}/{rest.split("/", 1)[0]}'
    if slug == 'guava':
        rest = path.removeprefix('guava/src/com/google/common/')
        return f'{slug}/{rest.split("/", 1)[0]}'
    # gson: partition by Java package directories rather than exact source paths.
    sub = path.split('/src/main/java/', 1)[-1].split('/')
    group = '/'.join(sub[:min(len(sub)-1, 5)]) if len(sub) > 1 else 'root'
    return f'{slug}/{group}'


def _split_groups(keys: set[str], *, seed: str = 'sagallm-public-java-v1') -> dict[str, str]:
    """Stratified per-repository group assignment, so all three splits exist per source."""
    result = {}
    by_repo = {}
    for key in sorted(keys):
        by_repo.setdefault(key.split('/', 1)[0], []).append(key)
    for repo, names in by_repo.items():
        ranked = sorted(names, key=lambda x: sha256((seed + ':' + x).encode('utf8')))
        n = len(ranked)
        if n < 3:
            # Gson may have few production packages. Avoid making unverifiable claims.
            for key in ranked:
                result[key] = 'train'
            continue
        val = max(1, n // 10)
        test = max(1, n // 10)
        for i, key in enumerate(ranked):
            result[key] = 'validation' if i < val else ('test' if i < val+test else 'train')
    if not all(v in result.values() for v in ('train','validation','test')):
        raise ValueError('Source groups cannot form all splits')
    return result


def _iter_java_files(root: Path, spec: Upstream):
    # Use ONLY files tracked by the pinned commit; untracked files never enter a corpus.
    r = subprocess.run(['git', '-C', str(root), 'ls-files', '-z'], capture_output=True, timeout=15, check=True)
    for name in sorted(n for n in r.stdout.split(b'\0') if n.endswith(b'.java')):
        try:
            rel = name.decode('utf-8')
        except UnicodeError:
            continue
        if _is_source(rel, spec.slug):
            file = root / rel
            yield file, rel


def prepare_public_corpus(sources: str | Path, output: str | Path) -> dict:
    """Inspect all 4 pinned local checkouts and create license-preserving JSONL.

    The procedure never downloads files, runs Java code, or executes project
    build scripts. The caller obtains each repository at the pinned commit.
    """
    source_root, out = Path(sources), Path(output)
    if out.exists() or out.is_symlink():
        raise FileExistsError('Refuse to overwrite corpus')
    if not source_root.is_dir() or source_root.is_symlink():
        raise ValueError('Expected an existing sources parent directory')
    recs, licenses, rejected = [], {}, Counter()
    unique = set(); total_bytes = 0
    for spec in UPSTREAMS:
        root = source_root / spec.slug
        raw_license = verify_upstream(root, spec)
        licenses[spec.slug] = raw_license
        count = 0
        for file, rel in _iter_java_files(root, spec):
            count += 1
            if count > MAX_FILES:
                raise ValueError('Java file limit exceeded')
            if file.is_symlink() or not file.is_file():
                rejected['unsafe_file'] += 1; continue
            size = file.stat().st_size
            if not 100 <= size <= MAX_BYTES_PER_FILE:
                rejected['size'] += 1; continue
            total_bytes += size
            if total_bytes > MAX_TOTAL_BYTES:
                raise ValueError('Total source size cap exceeded')
            data = file.read_bytes()
            if b'\0' in data:
                rejected['binary'] += 1; continue
            try:
                code = data.decode('utf8').replace('\r\n','\n').replace('\r','\n')
            except UnicodeError:
                rejected['encoding'] += 1; continue
            if sensitive_reason(code) or low_quality(code):
                rejected['screened'] += 1; continue
            content_sha = sha256(code.encode('utf8'))
            if content_sha in unique:
                rejected['duplicate'] += 1; continue
            unique.add(content_sha)
            recs.append({'repo':spec.slug,'path':rel,'group':category_key(spec.slug,rel),
                         'sha256':content_sha,'text':code})
    if len(recs) < 12:
        raise ValueError('Not enough verified Java files')
    group_splits = _split_groups({x['group'] for x in recs})
    for rec in recs:
        rec['split'] = group_splits[rec['group']]
    counts = Counter(x['split'] for x in recs)
    if not all(counts[k] > 0 for k in ('train','validation','test')):
        raise ValueError('Source groups result in an empty split')
    out.mkdir(parents=True)
    try:
        (out/'LICENSES').mkdir()
        for spec in UPSTREAMS:
            (out/'LICENSES'/f'{spec.slug}.txt').write_bytes(licenses[spec.slug])
        for split in ('train','validation','test'):
            with (out/f'{split}.jsonl').open('w',encoding='utf8') as f:
                for rec in recs:
                    if rec['split'] == split:
                        data={k:rec[k] for k in ('repo','path','group','sha256','text')}
                        f.write(json.dumps(data,sort_keys=True,ensure_ascii=False)+'\n')
        sources_manifest=[]
        for spec in UPSTREAMS:
            sources_manifest.append({'name':spec.slug,'url':spec.url,'revision':spec.revision,
                                     'license_id':spec.license_id,'license_blob':spec.license_blob,
                                     'license_file':f'LICENSES/{spec.slug}.txt',
                                     'license_sha256':sha256(licenses[spec.slug])})
        manifest={'format':FORMAT,'sources':sources_manifest,'splits':dict(counts),
                  'categories':{s:len({x['group'] for x in recs if x['split']==s}) for s in counts},
                  'total_source_bytes_seen':total_bytes,'rejects':dict(rejected),
                  'source_code_license_review':'PINNED_MIT_APACHE2_TEXT_HASH_CHECKED_NOT_LEGAL_ADVICE',
                  'training_use':'ONE_PASS_LANGUAGE_MODELING_ONLY',
                  'raw_code_licensing':'Attribution/notice files retained separately; individual upstream rights may still require review.',
                  'file_hashes':{s:sha256((out/f'{s}.jsonl').read_bytes()) for s in ('train','validation','test')}}
        (out/'manifest.json').write_text(json.dumps(manifest,sort_keys=True,indent=2)+'\n',encoding='utf8')
        return manifest
    except BaseException:
        import shutil
        shutil.rmtree(out)
        raise


def verify_public_corpus(corpus: str | Path) -> dict:
    root=Path(corpus)
    if not root.is_dir() or root.is_symlink():
        raise ValueError('Invalid corpus root')
    manifest=json.loads((root/'manifest.json').read_text(encoding='utf8'))
    if manifest.get('format') != FORMAT or not isinstance(manifest.get('file_hashes'),dict):
        raise ValueError('Unsupported public corpus manifest')
    if (len(manifest.get('sources',[])) != len(UPSTREAMS) or
        {s['name'] for s in manifest.get('sources',[])} != {s.slug for s in UPSTREAMS}):
        raise ValueError('Source inventory mismatch')
    for spec in UPSTREAMS:
        card=next(s for s in manifest['sources'] if s['name']==spec.slug)
        if any(card.get(k)!=getattr(spec,k) for k in ('url','revision','license_id','license_blob')):
            raise ValueError('Unexpected pinned source metadata')
        lic=root/'LICENSES'/f'{spec.slug}.txt'
        if (lic.is_symlink() or card.get('license_sha256')!=sha256(lic.read_bytes()) or
            git_blob_sha(lic.read_bytes())!=spec.license_blob):
            raise ValueError('License notice tampering')
    groups={}; unique=set(); unique_paths=set(); counts=Counter(); files={}
    for split in ('train','validation','test'):
        file=root/f'{split}.jsonl'
        if file.is_symlink() or not file.is_file() or sha256(file.read_bytes())!=manifest['file_hashes'].get(split):
            raise ValueError('Corpus data SHA-256 mismatch')
        all_rows=[]
        for line in file.read_text(encoding='utf8').splitlines():
            row=json.loads(line)
            if not all(isinstance(row.get(k),str) for k in ('repo','group','path','text','sha256')):
                raise ValueError('Invalid Java source record')
            if (row['repo'] not in {s.slug for s in UPSTREAMS} or
                not _is_source(row['path'],row['repo']) or
                row['group']!=category_key(row['repo'],row['path'])):
                raise ValueError('Unexpected Java provenance group')
            if sha256(row['text'].encode('utf8'))!=row['sha256']:
                raise ValueError('Java source code hash mismatch')
            if row['sha256'] in unique:
                raise ValueError('Identical Java source in multiple records')
            unique.add(row['sha256'])
            path_key=(row['repo'],row['path'])
            if path_key in unique_paths:
                raise ValueError('Duplicate Java source path')
            unique_paths.add(path_key)
            if row['group'] in groups and groups[row['group']]!=split:
                raise ValueError('Cross-split package/category contamination')
            groups[row['group']]=split
            counts[split]+=1
            all_rows.append(row)
        files[split]=all_rows
    if dict(counts)!=manifest.get('splits'):
        raise ValueError('Corpus split counts mismatch')
    if _split_groups(set(groups)) != groups:
        raise ValueError('Corpus split mapping differs from the reproducible policy')
    return {'counts':dict(counts),'total':sum(counts.values()),
            'train_file_sha256':manifest['file_hashes']['train'],
            'sources':[x['name'] for x in manifest['sources']], 'files':files}


if __name__=='__main__':
    import argparse
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--sources',type=Path)
    p.add_argument('--output',type=Path)
    p.add_argument('--verify',type=Path)
    args=p.parse_args()
    if args.verify:
        report=verify_public_corpus(args.verify)
        print(json.dumps({k:v for k,v in report.items() if k!='files'},indent=2))
    else:
        if args.sources is None or args.output is None:p.error('Provide --sources and --output')
        print(json.dumps(prepare_public_corpus(args.sources,args.output),indent=2))