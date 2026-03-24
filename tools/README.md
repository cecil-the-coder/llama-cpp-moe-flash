# Tools

## split_experts.py

**Status**: Not yet written (Phase 1 task P1.2)

Offline tool that reformats GGUF MoE expert tensors into per-expert binary files suitable
for io_uring streaming reads.

### Planned usage

```bash
python3 split_experts.py \
  --gguf /models/qwen3-235b-a22b/Q2_K/Qwen3-235B-A22B-Q2_K-00001-of-00002.gguf \
  --gguf /models/qwen3-235b-a22b/Q2_K/Qwen3-235B-A22B-Q2_K-00002-of-00002.gguf \
  --out  /models/qwen3-235b-a22b/experts/ \
  --align 2097152  # 2 MB alignment for optimal DMA
```

### Output layout

```
experts/
  manifest.json                      # layer/expert → file + offset + size
  L00_E0000.bin                      # layer 0, expert 0: gate_up + down packed
  L00_E0001.bin
  ...
  L93_E0127.bin                      # layer 93, expert 127
```

### Implementation notes

- Use `gguf` Python library (`pip install gguf`) to read tensor metadata
- Expert tensors are named `blk.{layer}.ffn_gate_up_exps` and `blk.{layer}.ffn_down_exps`
  (or `ffn_gate_exps` + `ffn_up_exps` if not merged)
- Expert index is dimension 2 (`ne[2]`) — slice at stride `ne[0] * ne[1] * type_size`
- Pad each output file to the next 2 MB boundary with zeros
- Verify: re-reading the files and stacking should exactly reproduce the original tensor

### Dependencies

```bash
pip install gguf numpy
```

## validate_split.py

**Status**: Not yet written (Phase 1 task P1.3)

Validates that a split expert directory round-trips cleanly against the source GGUF.

```bash
python3 validate_split.py \
  --gguf /models/qwen3-235b-a22b/Q2_K/Qwen3-235B-A22B-Q2_K-00001-of-00002.gguf \
  --experts /models/qwen3-235b-a22b/experts/
```
