#!/usr/bin/env python3
"""Dump GPT-2 weights to a flat binary file consumable by the C++ cached
inference engine in include/fdeep/llm/gpt2_cached.hpp.

The binary layout is intentionally simple:

  Header (104 bytes, little-endian):
    int32  magic           = 0x47505432 ("GPT2")
    int32  version         = 1
    int32  num_layers
    int32  hidden_dim
    int32  num_heads
    int32  head_dim
    int32  intermediate_dim
    int32  vocab_size
    int32  max_position
    int32  reserved        = 0
    float  layer_norm_epsilon
    char[64] preset_name (NUL-padded)

  Body (all float32, little-endian, contiguous, native row-major layouts):
    token_embedding:    [vocab_size, hidden_dim]
    position_embedding: [max_position, hidden_dim]
    for layer in 0..num_layers-1:
        attn_norm_gamma:    [hidden_dim]
        attn_norm_beta:     [hidden_dim]
        attn_q_kernel:      [hidden_dim, num_heads, head_dim]
        attn_q_bias:        [num_heads, head_dim]
        attn_k_kernel:      [hidden_dim, num_heads, head_dim]
        attn_k_bias:        [num_heads, head_dim]
        attn_v_kernel:      [hidden_dim, num_heads, head_dim]
        attn_v_bias:        [num_heads, head_dim]
        attn_o_kernel:      [num_heads, head_dim, hidden_dim]
        attn_o_bias:        [hidden_dim]
        ffn_norm_gamma:     [hidden_dim]
        ffn_norm_beta:      [hidden_dim]
        ffn1_kernel:        [hidden_dim, intermediate_dim]
        ffn1_bias:          [intermediate_dim]
        ffn2_kernel:        [intermediate_dim, hidden_dim]
        ffn2_bias:          [hidden_dim]
    final_norm_gamma:   [hidden_dim]
    final_norm_beta:    [hidden_dim]
    lm_head_kernel:     [hidden_dim, vocab_size]   (tied = token_embedding.T)
"""
import argparse
import os
import struct
import sys

os.environ.setdefault("CUDA_VISIBLE_DEVICES", "")
os.environ.setdefault("KERAS_BACKEND", "tensorflow")

import numpy as np
import keras
import keras_hub  # noqa: F401  (registers GPT2 layers)

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from convert_model import gelu_approximate  # noqa: F401  (registers serializable)
from save_gpt2_backbone_for_fdeep import (
    build_gpt2_like, copy_weights_from_keras_hub,
)


MAGIC = 0x47505432
VERSION = 1


def write_f32(f, arr: np.ndarray) -> None:
    a = np.ascontiguousarray(arr, dtype=np.float32)
    f.write(a.tobytes())


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--preset", default="gpt2_base_en")
    parser.add_argument("--output", required=True,
                        help="Path to write the binary weights file.")
    args = parser.parse_args()

    print(f"Loading keras_hub preset {args.preset!r}...")
    gpt2_lm = keras_hub.models.GPT2CausalLM.from_preset(args.preset)
    src = gpt2_lm.backbone
    cfg = src.get_config()
    vocab_size = cfg["vocabulary_size"]
    hidden_dim = cfg["hidden_dim"]
    num_heads = cfg["num_heads"]
    intermediate_dim = cfg["intermediate_dim"]
    num_layers = cfg["num_layers"]
    max_position = cfg["max_sequence_length"]
    layer_norm_epsilon = float(cfg.get("layer_norm_epsilon", 1e-5))
    head_dim = hidden_dim // num_heads

    # Build a small (seq_len=2) plain-Keras copy purely as a vehicle for
    # ``copy_weights_from_keras_hub`` -- only the weights are exported.
    dst = build_gpt2_like(seq_len=2, vocab_size=vocab_size,
                          max_position=max_position, hidden_dim=hidden_dim,
                          num_heads=num_heads, intermediate_dim=intermediate_dim,
                          num_layers=num_layers,
                          layer_norm_epsilon=layer_norm_epsilon,
                          with_lm_head=True)
    copy_weights_from_keras_hub(src, dst, num_layers, with_lm_head=True)
    print("Weights copied. Writing binary...")

    with open(args.output, "wb") as f:
        # Header
        preset_bytes = args.preset.encode("utf-8")[:64].ljust(64, b"\0")
        f.write(struct.pack(
            "<10i f 64s",
            MAGIC, VERSION, num_layers, hidden_dim, num_heads, head_dim,
            intermediate_dim, vocab_size, max_position, 0,
            layer_norm_epsilon, preset_bytes,
        ))
        # Embeddings
        token_w = dst.get_layer("token_embedding").get_weights()[0]
        write_f32(f, token_w)
        pos_w = dst.get_layer("position_embedding").get_weights()[0]
        write_f32(f, pos_w)

        for i in range(num_layers):
            attn_norm = dst.get_layer(f"block_{i}_attn_norm").get_weights()
            assert len(attn_norm) == 2
            write_f32(f, attn_norm[0]); write_f32(f, attn_norm[1])

            attn = dst.get_layer(f"block_{i}_attn").get_weights()
            # Keras MultiHeadAttention.get_weights returns (per Keras source,
            # in the order they were created):
            #   query_kernel, query_bias,
            #   key_kernel, key_bias,
            #   value_kernel, value_bias,
            #   output_kernel, output_bias
            # which matches the layout we declare in the binary.
            assert len(attn) == 8
            for w in attn:
                write_f32(f, w)

            ffn_norm = dst.get_layer(f"block_{i}_ffn_norm").get_weights()
            assert len(ffn_norm) == 2
            write_f32(f, ffn_norm[0]); write_f32(f, ffn_norm[1])

            ffn1 = dst.get_layer(f"block_{i}_ffn_intermediate").get_weights()
            assert len(ffn1) == 2
            write_f32(f, ffn1[0]); write_f32(f, ffn1[1])
            ffn2 = dst.get_layer(f"block_{i}_ffn_output").get_weights()
            assert len(ffn2) == 2
            write_f32(f, ffn2[0]); write_f32(f, ffn2[1])

        fn = dst.get_layer("final_norm").get_weights()
        assert len(fn) == 2
        write_f32(f, fn[0]); write_f32(f, fn[1])

        lm = dst.get_layer("lm_head").get_weights()
        assert len(lm) == 1
        write_f32(f, lm[0])

    size = os.path.getsize(args.output)
    print(f"Wrote {size:,} bytes to {args.output}")


if __name__ == "__main__":
    main()
