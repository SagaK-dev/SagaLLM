# Tokenizer backends

SagaLLM supports two **independently trained**, incompatible byte-level vocabularies:

| Backend | File `type` | Intended scope | Dependency |
| --- | --- | --- | --- |
| `python` | `byte_bpe_v1` | Educational, tiny offline corpora | Standard library |
| `hf` | `hf_byte_bpe_v1` | Larger multilingual corpora; Rust-backed BPE training | `tokenizers>=0.20,<1.0` (optional) |

Both preserve UTF-8 text through encode/decode and reserve `<pad>=0`, `<bos>=1`, `<eos>=2`, `<unk>=3`. **Other token IDs differ.** Never change the tokenizer used for a token dataset or checkpoint. SagaLLM checks the tokenizer file SHA-256 in training, resume, validation and inference when recorded.

## Local usage

```bash
python -m pip install -e '.[fast-tokenizer]'
sagallm tokenizer train --backend hf --data examples/toy_corpus.txt --output data/fast.json --vocab-size 512
sagallm tokenizer check --tokenizer data/fast.json --text '日本語, English, and C++ 🚀'
```

The `--backend python` default preserves older CLI behavior. `--backend hf` passes a document iterator into the native library, avoiding a Python list of all documents. The native trainer **still requires memory**, and no corpus-size or peak-RAM guarantee is made. PyTorch/CPU smoke tests run even when the optional dependency is missing; in GitHub Actions, the fast-backend test is installed and exercised.

## Research caveats

- The basic byte-level BPE pretokenizer uses ByteLevel conventions and no custom Japanese morphological preprocessing. Study Japanese/code token-per-character and token-per-byte rates on *licensed* representative corpora before selecting vocabulary size.
- Production choices such as byte-fallback, vocabulary shrinkage and chat special tokens need dedicated evaluation; this release is not a research-validated tokenizer.
- The saved `hf_byte_bpe_v1` JSON embeds the Rust tokenizer model in the same file. This keeps the existing one-file SHA-256 provenance checks intact.
- Installing optional dependencies may require network access; the project never downloads training data or pretrained weights automatically.