# Decisions

A running log of the choices made while building **Inside the LLM**, written as they were
made. Dates are the day the decision landed.

---

## 2026-10-28 — The model spike (the thing that had to work first)

**Problem.** Every view in this app needs model internals: attention weights per
layer and head, the residual stream per layer, and the next-token distribution. Those are
exactly the tensors that production inference stacks throw away.

**What I checked.** The ready-made ONNX checkpoints (`Xenova/distilgpt2`,
`onnx-community/gpt2`) expose `logits` and the KV cache, nothing else. transformers.js has
no `output_attentions` switch, because the information never leaves the ONNX graph — there
is no output node for it. So the off-the-shelf path was a dead end, as expected.

**Decision.** Export DistilGPT-2 myself, with the internals as first-class graph outputs:
[`scripts/export_model.py`](scripts/export_model.py). The wrapper module runs GPT-2 with
`attn_implementation="eager"` (PyTorch's fused SDPA kernel does not return attention
weights) and `output_attentions=True, output_hidden_states=True`, then reduces everything
to small, ready-to-plot tensors **inside the graph**:

| output           | shape             | what the UI does with it                    |
| ---------------- | ----------------- | ------------------------------------------- |
| `logits`         | `(1, V)`          | sampling + the live probability bar chart   |
| `attentions`     | `(1, L, H, T, T)` | head-grid heatmaps and the arc diagram      |
| `hidden_norms`   | `(1, L+1, T)`     | residual-stream magnitude strip             |
| `token_logprobs` | `(1, T)`          | colours the prompt tokens by model surprise |
| `lens_values`    | `(1, L+1, K)`     | logit lens: top-k log-probs per layer       |
| `lens_indices`   | `(1, L+1, K)`     | logit lens: top-k token ids per layer       |

Doing the top-k reduction in the graph matters: the raw logit lens is
`(L+1) x 50257` floats per step. Reduced to `K=10` it is 70 numbers, so the worker hands
the UI something it can plot directly instead of 1.4 MB of logits to sort on the main
thread.

**Verified before any UI existed** (`python scripts/export_model.py`, output recorded in
`model/verify.json`):

```
logits           (1, 50257) float32
attentions       (1, 6, 12, 5, 5) float32
hidden_norms     (1, 7, 5) float32
token_logprobs   (1, 5) float32
lens_values      (1, 7, 10) float32
lens_indices     (1, 7, 10) int64
attention rows sum to 1 (min 1.0000, max 1.0000) OK
```

Attention rows summing to exactly 1 is the proof that these are real post-softmax
attention weights and not something the exporter invented.

### Sub-decisions that came out of the spike

- **`dynamo=True`.** PyTorch 2.14's legacy TorchScript ONNX exporter dies on this graph
  (`RuntimeError: unordered_map::at: key not found`, inside functorch). The `torch.export`
  based exporter handles it without complaint.
- **No KV cache.** The model re-reads the whole sequence on every step. That is
  quadratic, and for the sequence lengths this demo allows (prompt + 32 tokens) it costs
  a few milliseconds, while buying something valuable: the full `T x T` attention matrix
  at every step, with no cache-shape gymnastics. A cached export would only ever return
  the last attention _row_, which kills the arc diagram.
- **The logit lens does not double-normalise.** HF's GPT-2 has already pushed
  `hidden_states[-1]` through `ln_f`, so the export applies `ln_f` to layers `0..L-1` and
  passes the last state through untouched. Getting this wrong makes the final layer's lens
  read-out subtly disagree with `logits`.
- **int8 dynamic quantization over `MatMul` _and_ `Gather`.** DistilGPT-2 is 82 M
  parameters, of which 38.6 M are the token embedding, which the LM head reuses (tied
  weights). Quantizing only `MatMul` leaves that embedding in fp32 and the download stays
  above 150 MB. Including `Gather` brings the whole model to **89 MB**, under the 100 MB
  target.

---

## 2026-10-28 — onnxruntime-web for inference, transformers.js for tokenization

The custom graph has six outputs and one input; it is not an `AutoModelForCausalLM` that
transformers.js knows how to drive. So inference talks to **onnxruntime-web** directly
(WebGPU execution provider, WASM fallback), which also means full control over how tensors
are read back.

Tokenization still goes through **`@huggingface/transformers`** (`AutoTokenizer`), loading
`tokenizer.json` from the app's own origin. Hand-rolling byte-level BPE to save a
dependency would have been a correctness risk for no user-visible benefit.

---

## 2026-10-28 — Model hosting

No Hugging Face token is available in this environment, so the 89 MB `model.onnx` ships as
a **GitHub Release asset** rather than being committed (`.gitignore` excludes `*.onnx`).
The worker fetches it with a streaming reader for progress, then stores it in the **Cache
API** so a second visit starts instantly. `VITE_MODEL_URL` overrides the location, which is
how you would point the app at a Hugging Face repo, R2 bucket or your own server instead.

---

## Open items and known limitations

See the bottom of this file — updated at the end of the build.
