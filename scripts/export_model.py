#!/usr/bin/env python3
"""Export DistilGPT-2 to ONNX with the extra outputs the visualizations need.

Vanilla ONNX checkpoints of GPT-2 only expose `logits`. "Inside the LLM" needs the
internals: attention weights per layer/head, a logit-lens read-out per layer and the
probability the model assigned to every token of the prompt. Those are produced here,
inside the graph, so the browser only ever receives small, ready-to-plot tensors.

Outputs of the exported graph (batch is fixed at 1):

    logits          (1, V)          next-token logits at the last position
    attentions      (1, L, H, T, T) softmaxed attention weights, every layer and head
    hidden_norms    (1, L+1, T)     L2 norm of each residual-stream state
    token_logprobs  (1, T)          log p(token_t | token_<t); position 0 is 0.0
    lens_values     (1, L+1, K)     logit-lens top-k log-probs, per layer
    lens_indices    (1, L+1, K)     logit-lens top-k token ids, per layer

Usage:
    python scripts/export_model.py --out public/model
    python scripts/export_model.py --out public/model --skip-quantize
"""

from __future__ import annotations

import argparse
import json
import shutil
import sys
from pathlib import Path

import torch
from torch import nn

MODEL_ID = "distilgpt2"
OPSET = 17
TOPK = 10


class InsideGPT2(nn.Module):
    """Wraps GPT-2 so that one forward pass returns everything the UI plots."""

    def __init__(self, model: nn.Module, topk: int = TOPK) -> None:
        super().__init__()
        self.m = model
        self.topk = topk

    def forward(self, input_ids: torch.Tensor):
        # int32 at the graph boundary: onnxruntime-web's WebGPU execution provider has
        # patchy int64 support, so the only int64 inside the graph is what GPT-2 itself needs.
        ids = input_ids.to(torch.int64)
        out = self.m(
            input_ids=ids,
            output_attentions=True,
            output_hidden_states=True,
            use_cache=False,
        )

        logits = out.logits  # (1, T, V)
        last_logits = logits[:, -1, :]  # (1, V)

        # Teacher-forced log-prob of every prompt token, so the token strip can colour
        # the prompt by "how surprising was this token to the model".
        shifted = torch.log_softmax(logits[:, :-1, :], dim=-1)  # (1, T-1, V)
        target = ids[:, 1:].unsqueeze(-1)  # (1, T-1, 1)
        token_logprobs = torch.gather(shifted, -1, target).squeeze(-1)  # (1, T-1)
        pad = torch.zeros(1, 1, dtype=token_logprobs.dtype, device=token_logprobs.device)
        token_logprobs = torch.cat([pad, token_logprobs], dim=1)  # (1, T)

        attentions = torch.stack(out.attentions, dim=1)  # (1, L, H, T, T)

        hidden = torch.stack(out.hidden_states, dim=1)  # (1, L+1, T, D)
        hidden_norms = hidden.norm(dim=-1)  # (1, L+1, T)

        # Logit lens: project every layer's residual stream at the last position through
        # ln_f + the (tied) LM head. hidden_states[-1] has already been through ln_f in
        # HF's GPT-2, so it is passed straight on instead of being normalised twice.
        states = list(out.hidden_states)
        ln_f = self.m.transformer.ln_f
        lensed = [ln_f(h[:, -1, :]) for h in states[:-1]] + [states[-1][:, -1, :]]
        lens_logprobs = torch.log_softmax(self.m.lm_head(torch.stack(lensed, dim=1)), dim=-1)
        lens_values, lens_indices = lens_logprobs.topk(self.topk, dim=-1)

        return (
            last_logits,
            attentions,
            hidden_norms,
            token_logprobs,
            lens_values,
            lens_indices.to(torch.int32),
        )


OUTPUT_NAMES = [
    "logits",
    "attentions",
    "hidden_norms",
    "token_logprobs",
    "lens_values",
    "lens_indices",
]


def export_fp32(out_dir: Path) -> Path:
    from transformers import AutoConfig, AutoModelForCausalLM, AutoTokenizer

    print(f"[1/4] loading {MODEL_ID} (eager attention, so attentions are returned)")
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_ID, attn_implementation="eager", dtype=torch.float32
    )
    model.eval()
    config = AutoConfig.from_pretrained(MODEL_ID)
    tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)

    wrapped = InsideGPT2(model).eval()
    example = torch.tensor([[464, 3139, 286, 4881, 318]], dtype=torch.int32)

    with torch.no_grad():
        reference = wrapped(example)

    fp32_path = out_dir / "model.fp32.onnx"
    print(f"[2/4] exporting ONNX (opset {OPSET}) -> {fp32_path.name}")
    torch.onnx.export(
        wrapped,
        (example,),
        str(fp32_path),
        input_names=["input_ids"],
        output_names=OUTPUT_NAMES,
        dynamic_axes={
            "input_ids": {1: "seq"},
            "attentions": {3: "seq", 4: "seq"},
            "hidden_norms": {2: "seq"},
            "token_logprobs": {1: "seq"},
        },
        opset_version=OPSET,
        do_constant_folding=True,
        dynamo=True,
    )

    meta = {
        "modelId": MODEL_ID,
        "numLayers": config.n_layer,
        "numHeads": config.n_head,
        "hiddenSize": config.n_embd,
        "vocabSize": config.vocab_size,
        "maxPositions": config.n_positions,
        "lensTopK": TOPK,
        "opset": OPSET,
        "outputs": OUTPUT_NAMES,
        "shapes": {name: list(t.shape) for name, t in zip(OUTPUT_NAMES, reference)},
    }
    (out_dir / "model-meta.json").write_text(json.dumps(meta, indent=2) + "\n")

    tokenizer_dir = out_dir / "tokenizer"
    tokenizer_dir.mkdir(parents=True, exist_ok=True)
    tokenizer.save_pretrained(tokenizer_dir)
    for junk in ("special_tokens_map.json",):
        leftover = tokenizer_dir / junk
        if leftover.exists():
            leftover.unlink()
    (tokenizer_dir / "config.json").write_text(config.to_json_string())
    print(f"       tokenizer + config written to {tokenizer_dir}")

    return fp32_path


def quantize(fp32_path: Path, out_dir: Path) -> Path:
    from onnxruntime.quantization import QuantType, quantize_dynamic

    int8_path = out_dir / "model.int8.onnx"
    print("[3/4] dynamic int8 quantization (MatMul + Gather, so the tied embedding shrinks too)")
    quantize_dynamic(
        model_input=str(fp32_path),
        model_output=str(int8_path),
        weight_type=QuantType.QInt8,
        op_types_to_quantize=["MatMul", "Gather"],
        extra_options={"MatMulConstBOnly": True},
    )
    return int8_path


def verify(model_path: Path, out_dir: Path) -> None:
    import numpy as np
    import onnxruntime as ort

    print(f"[4/4] verifying {model_path.name} with onnxruntime")
    session = ort.InferenceSession(str(model_path), providers=["CPUExecutionProvider"])
    ids = np.array([[464, 3139, 286, 4881, 318]], dtype=np.int32)
    result = session.run(None, {"input_ids": ids})
    named = dict(zip([o.name for o in session.get_outputs()], result))

    for name, value in named.items():
        print(f"       {name:<16} {tuple(value.shape)} {value.dtype}")

    attn = named["attentions"]
    rows = attn.sum(axis=-1)
    assert np.allclose(rows, 1.0, atol=1e-2), f"attention rows do not sum to 1: {rows.min()}..{rows.max()}"
    print(f"       attention rows sum to 1 (min {rows.min():.4f}, max {rows.max():.4f}) OK")

    top = named["logits"][0].argsort()[::-1][:5]
    print(f"       top-5 next-token ids for 'The capital of France is': {top.tolist()}")
    print(f"       logit-lens top-1 per layer: {named['lens_indices'][0][:, 0].tolist()}")

    size_mb = model_path.stat().st_size / 1e6
    print(f"       size: {size_mb:.1f} MB")
    (out_dir / "verify.json").write_text(
        json.dumps(
            {
                "model": model_path.name,
                "sizeMB": round(size_mb, 1),
                "shapes": {k: list(v.shape) for k, v in named.items()},
                "topNextTokenIds": top.tolist(),
                "lensTop1PerLayer": named["lens_indices"][0][:, 0].tolist(),
            },
            indent=2,
        )
        + "\n"
    )


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", default="public/model", help="output directory")
    parser.add_argument("--skip-quantize", action="store_true")
    parser.add_argument("--keep-fp32", action="store_true", help="do not delete the fp32 export")
    args = parser.parse_args()

    out_dir = Path(args.out).resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    fp32_path = export_fp32(out_dir)
    print(f"       fp32 size: {fp32_path.stat().st_size / 1e6:.1f} MB")

    final = fp32_path
    if not args.skip_quantize:
        final = quantize(fp32_path, out_dir)

    verify(final, out_dir)

    target = out_dir / "model.onnx"
    shutil.copyfile(final, target)
    print(f"done -> {target} ({target.stat().st_size / 1e6:.1f} MB)")

    if not args.keep_fp32 and final is not fp32_path:
        # The dynamo exporter writes weights next to the graph as external data.
        fp32_path.unlink(missing_ok=True)
        Path(str(fp32_path) + ".data").unlink(missing_ok=True)
        (out_dir / "model.int8.onnx").unlink(missing_ok=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
