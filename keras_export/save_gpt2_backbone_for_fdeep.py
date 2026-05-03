#!/usr/bin/env python3
"""Rebuild a keras_hub GPT-2 model as a plain Keras Functional model
using only layer types that frugally-deep supports, copy weights from
the keras_hub model, and save it in .keras format.

The resulting model takes:
  - token_ids: int32 of shape (batch, seq_len)
  - position_ids: int32 of shape (batch, seq_len)

and outputs either:
  - the final hidden state of shape (batch, seq_len, hidden_dim) (default), or
  - vocabulary logits of shape (batch, seq_len, vocab_size) when ``--with-lm-head``
    is set. The LM head is a tied projection: its kernel is the token-embedding
    matrix transposed, matching keras_hub's GPT2CausalLM behaviour.

Causal attention is enabled via MultiHeadAttention(use_causal_mask=True).
GELU uses the tanh-approximation form (matching keras_hub's GPT-2). The
``gelu_approximate`` helper from convert_model is registered as a
serializable Keras activation; the converter rewrites it to a plain
``gelu`` activation with ``approximate=True`` in the layer config, which
the C++ runtime understands.
"""

import argparse
import os
import sys

os.environ.setdefault("CUDA_VISIBLE_DEVICES", "")
os.environ.setdefault("KERAS_BACKEND", "tensorflow")

import keras
import keras_hub
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from convert_model import gelu_approximate  # noqa: E402  (registers serializable)


def build_gpt2_like(seq_len: int, vocab_size: int, max_position: int,
                    hidden_dim: int, num_heads: int, intermediate_dim: int,
                    num_layers: int, layer_norm_epsilon: float = 1e-5,
                    with_lm_head: bool = False) -> keras.Model:
    """Build a GPT-2-style decoder-only transformer with plain Keras layers."""
    head_dim = hidden_dim // num_heads

    token_ids = keras.Input(shape=(seq_len,), dtype="int32", name="token_ids")
    position_ids = keras.Input(shape=(seq_len,), dtype="int32", name="position_ids")

    tok_emb = keras.layers.Embedding(vocab_size, hidden_dim, name="token_embedding")(token_ids)
    pos_emb = keras.layers.Embedding(max_position, hidden_dim, name="position_embedding")(position_ids)
    x = keras.layers.Add(name="embeddings_add")([tok_emb, pos_emb])

    for i in range(num_layers):
        # Pre-norm self-attention
        n = keras.layers.LayerNormalization(epsilon=layer_norm_epsilon,
                                            name=f"block_{i}_attn_norm")(x)
        a = keras.layers.MultiHeadAttention(
            num_heads=num_heads, key_dim=head_dim, value_dim=head_dim,
            use_bias=True, name=f"block_{i}_attn")(n, n, n, use_causal_mask=True)
        x = keras.layers.Add(name=f"block_{i}_attn_residual")([x, a])

        # Pre-norm FFN
        n = keras.layers.LayerNormalization(epsilon=layer_norm_epsilon,
                                            name=f"block_{i}_ffn_norm")(x)
        n = keras.layers.Dense(intermediate_dim, activation=gelu_approximate,
                               name=f"block_{i}_ffn_intermediate")(n)
        n = keras.layers.Dense(hidden_dim, name=f"block_{i}_ffn_output")(n)
        x = keras.layers.Add(name=f"block_{i}_ffn_residual")([x, n])

    x = keras.layers.LayerNormalization(epsilon=layer_norm_epsilon, name="final_norm")(x)

    if with_lm_head:
        # GPT-2 uses a tied LM head: logits = hidden @ E.T, where E is the
        # token-embedding matrix. We materialise it as a plain bias-less Dense
        # so frugally-deep can run it without extra layer support.
        x = keras.layers.Dense(vocab_size, use_bias=False, name="lm_head")(x)

    return keras.Model(inputs=[token_ids, position_ids], outputs=x, name="gpt2_like")


def copy_weights_from_keras_hub(src_backbone, dst_model: keras.Model, num_layers: int,
                                 with_lm_head: bool = False) -> None:
    """Copy weights from a keras_hub GPT2Backbone into our plain-Keras model."""
    # Embeddings
    src_token = src_backbone.get_layer("token_embedding")
    embedding_matrix = src_token.embeddings.numpy()
    dst_model.get_layer("token_embedding").set_weights([embedding_matrix])
    if with_lm_head:
        # Tied weights: kernel is E.T (Dense maps hidden_dim -> vocab_size).
        dst_model.get_layer("lm_head").set_weights([embedding_matrix.T])

    src_pos = src_backbone.get_layer("position_embedding")
    # PositionEmbedding stores its lookup table under .position_embeddings
    pos_w = src_pos.position_embeddings.numpy()
    dst_model.get_layer("position_embedding").set_weights([pos_w])

    for i in range(num_layers):
        src_block = src_backbone.get_layer(f"transformer_layer_{i}")

        # Attention: keras_hub uses Keras MultiHeadAttention internally with
        # the same einsum-style weight layout.
        src_attn = src_block._self_attention_layer
        dst_attn = dst_model.get_layer(f"block_{i}_attn")
        dst_attn.set_weights(src_attn.get_weights())

        # Pre-attention LayerNorm
        src_attn_ln = src_block._self_attention_layer_norm
        dst_model.get_layer(f"block_{i}_attn_norm").set_weights(src_attn_ln.get_weights())

        # FFN dense layers
        src_ffn1 = src_block._feedforward_intermediate_dense
        src_ffn2 = src_block._feedforward_output_dense
        dst_model.get_layer(f"block_{i}_ffn_intermediate").set_weights(src_ffn1.get_weights())
        dst_model.get_layer(f"block_{i}_ffn_output").set_weights(src_ffn2.get_weights())

        # Pre-FFN LayerNorm
        src_ffn_ln = src_block._feedforward_layer_norm
        dst_model.get_layer(f"block_{i}_ffn_norm").set_weights(src_ffn_ln.get_weights())

    # Final LayerNorm
    dst_model.get_layer("final_norm").set_weights(src_backbone.get_layer("layer_norm").get_weights())


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--preset", default="gpt2_base_en")
    parser.add_argument("--seq-len", type=int, default=32)
    parser.add_argument("--output", required=True)
    parser.add_argument("--with-lm-head", action="store_true",
                        help="Append a tied LM head so the model outputs "
                             "vocabulary logits instead of hidden states.")
    parser.add_argument("--verify", action="store_true",
                        help="Run a forward pass through both models and "
                             "report the max absolute difference.")
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
    layer_norm_epsilon = cfg.get("layer_norm_epsilon", 1e-5)

    print(f"  vocab_size={vocab_size}, hidden_dim={hidden_dim}, "
          f"num_heads={num_heads}, intermediate_dim={intermediate_dim}, "
          f"num_layers={num_layers}, max_position={max_position}, "
          f"layer_norm_epsilon={layer_norm_epsilon}")

    print(f"Building plain-Keras GPT-2 with seq_len={args.seq_len}, "
          f"with_lm_head={args.with_lm_head}...")
    dst = build_gpt2_like(seq_len=args.seq_len, vocab_size=vocab_size,
                          max_position=max_position, hidden_dim=hidden_dim,
                          num_heads=num_heads, intermediate_dim=intermediate_dim,
                          num_layers=num_layers,
                          layer_norm_epsilon=layer_norm_epsilon,
                          with_lm_head=args.with_lm_head)
    print("Copying weights...")
    copy_weights_from_keras_hub(src, dst, num_layers,
                                 with_lm_head=args.with_lm_head)

    if args.verify:
        print("Verifying numerics against keras_hub reference...")
        rng = np.random.default_rng(0)
        token_ids = rng.integers(0, vocab_size, size=(1, args.seq_len), dtype=np.int32)
        position_ids = np.arange(args.seq_len, dtype=np.int32)[None, :]
        padding_mask = np.ones_like(token_ids, dtype=np.int32)

        if args.with_lm_head:
            # Compare against full GPT2CausalLM logits.
            ref = gpt2_lm({"token_ids": token_ids, "padding_mask": padding_mask}).numpy()
        else:
            ref = src({"token_ids": token_ids, "padding_mask": padding_mask}).numpy()
        ours = dst([token_ids, position_ids]).numpy()
        diff = np.max(np.abs(ref - ours))
        print(f"  max |ref - ours| = {diff:.3e}")
        if args.with_lm_head:
            # Logits scale much larger than hidden states (often 50–200), so
            # an absolute tolerance scaled by output magnitude is appropriate.
            ref_max = float(np.max(np.abs(ref)))
            print(f"  ref output scale (max |ref|) = {ref_max:.3f}")
            tol = max(1e-3 * ref_max, 1e-2)
        else:
            tol = 1e-3
        assert diff < tol, f"Numerics differ too much: {diff} (tol={tol})"
        print("  OK")

    print(f"Saving to {args.output}...")
    dst.save(args.output)
    print("Done.")


if __name__ == "__main__":
    main()
