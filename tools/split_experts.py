#!/usr/bin/env python3
"""Split MoE expert tensors from a GGUF file into per-expert binary files.

Each expert's gate, up, and down projection weights are extracted and packed
into a single aligned binary file for efficient io_uring reads.

Usage:
    python3 split_experts.py model.gguf [--output-dir experts/] [--align 2097152] [--verify]
"""

import argparse
import json
import mmap
import os
import struct
import sys
from dataclasses import dataclass, field
from pathlib import Path

# ── GGUF type definitions ──────────────────────────────────────────────────

# (block_size_elements, block_size_bytes)
GGML_TYPE_INFO: dict[int, tuple[int, int, str]] = {
    0:  (1,   4,   "F32"),
    1:  (1,   2,   "F16"),
    2:  (32,  18,  "Q4_0"),
    3:  (32,  20,  "Q4_1"),
    6:  (32,  20,  "Q5_0"),
    7:  (32,  24,  "Q5_1"),
    8:  (32,  34,  "Q8_0"),
    9:  (32,  36,  "Q8_1"),
    10: (256, 84,  "Q2_K"),
    11: (256, 110, "Q3_K"),
    12: (256, 144, "Q4_K"),
    13: (256, 176, "Q5_K"),
    14: (256, 210, "Q6_K"),
    15: (256, 292, "Q8_K"),
    16: (256, 66,  "IQ2_XXS"),
    17: (256, 74,  "IQ2_XS"),
    18: (256, 98,  "IQ3_XXS"),
    19: (256, 54,  "IQ1_S"),
    20: (32,  18,  "IQ4_NL"),
    21: (256, 110, "IQ3_S"),
    22: (256, 66,  "IQ2_S"),
    23: (256, 70,  "IQ4_XS"),
    24: (256, 56,  "IQ1_M"),
    25: (1,   2,   "BF16"),
    26: (32,  18,  "Q4_0_4_4"),
    27: (32,  18,  "Q4_0_4_8"),
    28: (32,  18,  "Q4_0_8_8"),
    30: (256, 54,  "TQ1_0"),
    31: (256, 66,  "TQ2_0"),
    32: (32,  18,  "IQ4_NL_4_4"),
}


def type_byte_size(ttype: int, n_elements: int) -> int:
    """Compute total byte size for n_elements of a given GGML type."""
    block_elems, block_bytes, _ = GGML_TYPE_INFO[ttype]
    assert n_elements % block_elems == 0, (
        f"n_elements={n_elements} not divisible by block_size={block_elems}"
    )
    return (n_elements // block_elems) * block_bytes


def type_name(ttype: int) -> str:
    return GGML_TYPE_INFO.get(ttype, (0, 0, f"type{ttype}"))[2]


# ── GGUF reader (minimal, header-only) ────────────────────────────────────

def _read_string(f) -> str:
    length = struct.unpack("<Q", f.read(8))[0]
    return f.read(length).decode("utf-8")


def _skip_value(f, vtype: int | None = None) -> None:
    """Skip a GGUF metadata value."""
    if vtype is None:
        vtype = struct.unpack("<I", f.read(4))[0]
    SIZES = {0:1, 1:1, 2:2, 3:2, 4:4, 5:4, 6:4, 7:1, 10:8, 11:8, 12:8, 13:4}
    if vtype == 8:  # string
        _read_string(f)
    elif vtype == 9:  # array
        atype = struct.unpack("<I", f.read(4))[0]
        alen = struct.unpack("<Q", f.read(8))[0]
        for _ in range(alen):
            _skip_value(f, atype)
    elif vtype in SIZES:
        f.read(SIZES[vtype])
    else:
        raise ValueError(f"Unknown GGUF value type {vtype}")


@dataclass
class TensorInfo:
    name: str
    dims: list[int]  # GGUF order (ne[0], ne[1], ...)
    ttype: int
    offset: int  # relative to data section start
    n_bytes: int = 0
    abs_offset: int = 0  # absolute file offset


@dataclass
class GGUFHeader:
    version: int
    n_tensors: int
    n_kv: int
    tensors: list[TensorInfo] = field(default_factory=list)
    data_start: int = 0
    alignment: int = 32


def read_gguf_header(path: str) -> GGUFHeader:
    """Read GGUF header and tensor metadata (no tensor data)."""
    f = open(path, "rb")
    magic = f.read(4)
    assert magic == b"GGUF", f"Not a GGUF file (magic={magic!r})"

    version = struct.unpack("<I", f.read(4))[0]
    n_tensors = struct.unpack("<Q", f.read(8))[0]
    n_kv = struct.unpack("<Q", f.read(8))[0]

    header = GGUFHeader(version=version, n_tensors=n_tensors, n_kv=n_kv)

    # Read KV pairs (looking for alignment override)
    for _ in range(n_kv):
        key = _read_string(f)
        if key == "general.alignment":
            vtype = struct.unpack("<I", f.read(4))[0]
            header.alignment = struct.unpack("<I", f.read(4))[0] if vtype == 4 else 32
        else:
            _skip_value(f)

    # Read tensor info
    for _ in range(n_tensors):
        name = _read_string(f)
        n_dims = struct.unpack("<I", f.read(4))[0]
        dims = [struct.unpack("<Q", f.read(8))[0] for _ in range(n_dims)]
        ttype = struct.unpack("<I", f.read(4))[0]
        offset = struct.unpack("<Q", f.read(8))[0]

        n_elements = 1
        for d in dims:
            n_elements *= d
        n_bytes = type_byte_size(ttype, n_elements)

        header.tensors.append(TensorInfo(
            name=name, dims=dims, ttype=ttype,
            offset=offset, n_bytes=n_bytes,
        ))

    # Data section starts at next alignment boundary after header
    header.data_start = (f.tell() + header.alignment - 1) & ~(header.alignment - 1)

    # Compute absolute offsets
    for t in header.tensors:
        t.abs_offset = header.data_start + t.offset

    f.close()
    return header


# ── Expert identification ─────────────────────────────────────────────────

@dataclass
class LayerExperts:
    """Expert tensors for one transformer layer."""
    layer_idx: int
    n_experts: int
    # Exactly one of these patterns is set:
    # Pattern A: separate gate + up + down
    gate_exps: TensorInfo | None = None
    up_exps: TensorInfo | None = None
    down_exps: TensorInfo | None = None
    # Pattern B: merged gate_up + down
    gate_up_exps: TensorInfo | None = None

    @property
    def is_merged(self) -> bool:
        return self.gate_up_exps is not None


def find_expert_layers(header: GGUFHeader) -> list[LayerExperts]:
    """Identify all MoE layers and their expert tensors."""
    # Group tensors by block index
    by_layer: dict[int, dict[str, TensorInfo]] = {}
    for t in header.tensors:
        # Match patterns like blk.N.ffn_*_exps.weight
        if "_exps" not in t.name:
            continue
        parts = t.name.split(".")
        # blk.N.ffn_XXX_exps.weight
        try:
            layer_idx = int(parts[1])
        except (IndexError, ValueError):
            continue
        if layer_idx not in by_layer:
            by_layer[layer_idx] = {}

        if "gate_up_exps" in t.name:
            by_layer[layer_idx]["gate_up"] = t
        elif "gate_exps" in t.name:
            by_layer[layer_idx]["gate"] = t
        elif "up_exps" in t.name:
            by_layer[layer_idx]["up"] = t
        elif "down_exps" in t.name:
            by_layer[layer_idx]["down"] = t

    layers = []
    for idx in sorted(by_layer):
        d = by_layer[idx]
        if "gate_up" in d:
            n_experts = d["gate_up"].dims[-1]
            layers.append(LayerExperts(
                layer_idx=idx, n_experts=n_experts,
                gate_up_exps=d["gate_up"],
                down_exps=d.get("down"),
            ))
        elif "gate" in d and "up" in d and "down" in d:
            n_experts = d["gate"].dims[-1]
            layers.append(LayerExperts(
                layer_idx=idx, n_experts=n_experts,
                gate_exps=d["gate"],
                up_exps=d["up"],
                down_exps=d["down"],
            ))
        elif "up" in d and "down" in d:
            # Some architectures have no gate (just up + down)
            n_experts = d["up"].dims[-1]
            layers.append(LayerExperts(
                layer_idx=idx, n_experts=n_experts,
                up_exps=d["up"],
                down_exps=d["down"],
            ))
        else:
            print(f"WARNING: layer {idx} has incomplete expert tensors: {list(d.keys())}", file=sys.stderr)

    return layers


# ── Splitting ─────────────────────────────────────────────────────────────

def expert_stride(t: TensorInfo) -> int:
    """Byte stride between consecutive experts in a 3D tensor [.., .., n_experts]."""
    n_experts = t.dims[-1]
    inner_elements = 1
    for d in t.dims[:-1]:
        inner_elements *= d
    return type_byte_size(t.ttype, inner_elements)


def split_experts(
    gguf_path: str,
    output_dir: str,
    alignment: int = 2 * 1024 * 1024,  # 2 MB default
    verify: bool = False,
) -> dict:
    """Split expert tensors into per-expert files.

    Returns the manifest dict.
    """
    header = read_gguf_header(gguf_path)
    layers = find_expert_layers(header)

    if not layers:
        print("ERROR: No MoE expert layers found in this model.", file=sys.stderr)
        sys.exit(1)

    print(f"Found {len(layers)} MoE layers, {layers[0].n_experts} experts each")
    print(f"Merged gate_up: {layers[0].is_merged}")

    os.makedirs(output_dir, exist_ok=True)

    # Memory-map the GGUF for efficient slicing
    fd = os.open(gguf_path, os.O_RDONLY)
    file_size = os.fstat(fd).st_size
    mm = mmap.mmap(fd, file_size, access=mmap.ACCESS_READ)

    manifest = {
        "version": 1,
        "source_gguf": os.path.basename(gguf_path),
        "alignment": alignment,
        "n_layers": len(layers),
        "n_experts": layers[0].n_experts,
        "layers": [],
    }

    total_written = 0

    for layer in layers:
        layer_manifest = {
            "layer": layer.layer_idx,
            "n_experts": layer.n_experts,
            "experts": [],
        }

        for e in range(layer.n_experts):
            # Collect the raw bytes for this expert
            parts: list[tuple[str, bytes, TensorInfo]] = []

            if layer.is_merged:
                # Pattern B: gate_up merged + down
                t = layer.gate_up_exps
                stride = expert_stride(t)
                start = t.abs_offset + e * stride
                parts.append(("gate_up", mm[start:start + stride], t))

                if layer.down_exps:
                    t = layer.down_exps
                    stride_d = expert_stride(t)
                    start = t.abs_offset + e * stride_d
                    parts.append(("down", mm[start:start + stride_d], t))
            else:
                # Pattern A: separate gate + up + down
                if layer.gate_exps:
                    t = layer.gate_exps
                    stride_g = expert_stride(t)
                    start = t.abs_offset + e * stride_g
                    parts.append(("gate", mm[start:start + stride_g], t))

                if layer.up_exps:
                    t = layer.up_exps
                    stride_u = expert_stride(t)
                    start = t.abs_offset + e * stride_u
                    parts.append(("up", mm[start:start + stride_u], t))

                if layer.down_exps:
                    t = layer.down_exps
                    stride_d = expert_stride(t)
                    start = t.abs_offset + e * stride_d
                    parts.append(("down", mm[start:start + stride_d], t))

            # Write combined expert file
            filename = f"L{layer.layer_idx:03d}_E{e:04d}.bin"
            filepath = os.path.join(output_dir, filename)

            expert_info = {
                "id": e,
                "file": filename,
                "parts": [],
            }

            offset_in_file = 0
            with open(filepath, "wb") as out:
                for part_name, data, tensor_info in parts:
                    out.write(data)
                    expert_info["parts"].append({
                        "name": part_name,
                        "offset": offset_in_file,
                        "size": len(data),
                        "type": type_name(tensor_info.ttype),
                        "dims": tensor_info.dims[:-1],  # per-expert dims (without expert axis)
                    })
                    offset_in_file += len(data)

                # Pad to alignment boundary
                unpadded_size = offset_in_file
                if alignment > 0:
                    padded_size = (unpadded_size + alignment - 1) & ~(alignment - 1)
                    pad_bytes = padded_size - unpadded_size
                    if pad_bytes > 0:
                        out.write(b"\x00" * pad_bytes)
                    expert_info["padded_size"] = padded_size
                else:
                    expert_info["padded_size"] = unpadded_size

                expert_info["raw_size"] = unpadded_size

            total_written += expert_info["padded_size"]
            layer_manifest["experts"].append(expert_info)

        manifest["layers"].append(layer_manifest)

        # Progress
        pct = (layer.layer_idx + 1) / len(layers) * 100
        exp0 = layer_manifest["experts"][0]
        print(
            f"  Layer {layer.layer_idx:3d}: {layer.n_experts} experts × "
            f"{exp0['raw_size']:,} bytes = {layer.n_experts * exp0['padded_size'] / 1024 / 1024:.1f} MB "
            f"({pct:.0f}%)"
        )

    # Write manifest
    manifest["total_bytes"] = total_written
    manifest_path = os.path.join(output_dir, "manifest.json")
    with open(manifest_path, "w") as f:
        json.dump(manifest, f, indent=2)

    print(f"\nWrote {total_written / 1024 / 1024 / 1024:.2f} GB to {output_dir}/")
    print(f"Manifest: {manifest_path}")

    # Verify round-trip if requested
    if verify:
        print("\nVerifying round-trip...")
        errors = verify_roundtrip(gguf_path, output_dir, manifest, mm)
        if errors:
            print(f"VERIFICATION FAILED: {errors} mismatches", file=sys.stderr)
            sys.exit(1)
        else:
            print("Verification PASSED: all expert data matches original")

    mm.close()
    os.close(fd)
    return manifest


# ── Verification ──────────────────────────────────────────────────────────

def verify_roundtrip(
    gguf_path: str,
    output_dir: str,
    manifest: dict,
    mm: mmap.mmap,
) -> int:
    """Verify that split files match the original GGUF tensor data."""
    header = read_gguf_header(gguf_path)
    layers = find_expert_layers(header)

    errors = 0
    for layer, layer_m in zip(layers, manifest["layers"]):
        for e in range(layer.n_experts):
            expert_m = layer_m["experts"][e]
            filepath = os.path.join(output_dir, expert_m["file"])

            with open(filepath, "rb") as f:
                split_data = f.read(expert_m["raw_size"])

            # Reconstruct what the original data should be
            original_parts = []
            if layer.is_merged:
                t = layer.gate_up_exps
                stride = expert_stride(t)
                start = t.abs_offset + e * stride
                original_parts.append(mm[start:start + stride])
                if layer.down_exps:
                    t = layer.down_exps
                    stride_d = expert_stride(t)
                    start = t.abs_offset + e * stride_d
                    original_parts.append(mm[start:start + stride_d])
            else:
                for tensor in [layer.gate_exps, layer.up_exps, layer.down_exps]:
                    if tensor:
                        s = expert_stride(tensor)
                        start = tensor.abs_offset + e * s
                        original_parts.append(mm[start:start + s])

            original = b"".join(original_parts)
            if split_data != original:
                print(
                    f"  MISMATCH: layer {layer.layer_idx} expert {e} "
                    f"(split={len(split_data)}, orig={len(original)})",
                    file=sys.stderr,
                )
                errors += 1

        if (layer.layer_idx + 1) % 10 == 0:
            print(f"  Verified layer {layer.layer_idx}")

    return errors


# ── CLI ───────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="Split MoE expert tensors from GGUF into per-expert files"
    )
    parser.add_argument("gguf", help="Path to GGUF model file")
    parser.add_argument(
        "--output-dir", "-o", default=None,
        help="Output directory (default: <gguf_dir>/experts/)",
    )
    parser.add_argument(
        "--align", type=int, default=2 * 1024 * 1024,
        help="Alignment padding in bytes (default: 2MB for DMA)",
    )
    parser.add_argument(
        "--verify", action="store_true",
        help="Verify round-trip after splitting",
    )
    parser.add_argument(
        "--info", action="store_true",
        help="Print tensor info without splitting",
    )
    args = parser.parse_args()

    if not os.path.exists(args.gguf):
        print(f"ERROR: File not found: {args.gguf}", file=sys.stderr)
        sys.exit(1)

    header = read_gguf_header(args.gguf)
    layers = find_expert_layers(header)

    if args.info:
        print(f"GGUF v{header.version}, {header.n_tensors} tensors")
        print(f"MoE layers: {len(layers)}")
        if layers:
            l0 = layers[0]
            print(f"Experts per layer: {l0.n_experts}")
            print(f"Merged gate_up: {l0.is_merged}")
            print(f"\nPer-expert sizes (layer {l0.layer_idx}):")
            if l0.is_merged:
                s = expert_stride(l0.gate_up_exps)
                print(f"  gate_up: {s:,} bytes ({type_name(l0.gate_up_exps.ttype)})")
            else:
                if l0.gate_exps:
                    s = expert_stride(l0.gate_exps)
                    print(f"  gate: {s:,} bytes ({type_name(l0.gate_exps.ttype)})")
                if l0.up_exps:
                    s = expert_stride(l0.up_exps)
                    print(f"  up:   {s:,} bytes ({type_name(l0.up_exps.ttype)})")
            if l0.down_exps:
                s = expert_stride(l0.down_exps)
                print(f"  down: {s:,} bytes ({type_name(l0.down_exps.ttype)})")

            # Total estimate
            per_expert = 0
            if l0.is_merged:
                per_expert += expert_stride(l0.gate_up_exps)
            else:
                if l0.gate_exps:
                    per_expert += expert_stride(l0.gate_exps)
                if l0.up_exps:
                    per_expert += expert_stride(l0.up_exps)
            if l0.down_exps:
                per_expert += expert_stride(l0.down_exps)

            padded = (per_expert + args.align - 1) & ~(args.align - 1) if args.align > 0 else per_expert
            total = len(layers) * l0.n_experts * padded
            print(f"\n  Combined per expert: {per_expert:,} bytes ({per_expert/1024:.1f} KB)")
            print(f"  Padded to {args.align}: {padded:,} bytes ({padded/1024/1024:.1f} MB)")
            print(f"  Total estimated: {total/1024/1024/1024:.2f} GB "
                  f"({len(layers)} layers × {l0.n_experts} experts × {padded:,} bytes)")
        return

    if args.output_dir is None:
        args.output_dir = os.path.join(os.path.dirname(args.gguf), "experts")

    split_experts(args.gguf, args.output_dir, args.align, args.verify)


if __name__ == "__main__":
    main()
