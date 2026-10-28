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

## 2026-10-28 — onnxruntime-web for inference, `@huggingface/tokenizers` for tokenization

The custom graph has six outputs and one input; it is not an `AutoModelForCausalLM` that
transformers.js knows how to drive. So inference talks to **onnxruntime-web** directly
(WebGPU execution provider, WASM fallback), which also means full control over how tensors
are read back.

The plan was to keep **`@huggingface/transformers`** for tokenization. Building it proved
that wrong: transformers.js bundles its own copy of onnxruntime-web, so importing it purely
for `AutoTokenizer` emitted a _second_ 55 MB set of WASM variants into `dist/`. The
tokenizer half of transformers.js ships separately as **`@huggingface/tokenizers`** — same
implementation, same maintainers, no inference runtime attached — and it returns the raw BPE
token strings (`Ġthe`, `Ċ`) that the token strip wants to show. Hand-rolling byte-level BPE
would have saved nothing and risked being subtly wrong.

### Keeping onnxruntime-web out of the bundle entirely

ORT's entry points reference their WASM binaries through `new URL(..., import.meta.url)`,
which any bundler follows, so a naive `import * as ort from 'onnxruntime-web/webgpu'` put
55 MB of `.wasm` into the build output. The worker now loads the prebuilt ESM bundle from
the app's own `public/ort/` at runtime, with a type-only import for the API. The build drops
to 275 kB of application code, and [`scripts/copy_ort.mjs`](scripts/copy_ort.mjs) copies
exactly the six files the two entry points import — so the deployed asset set is something
a person chose rather than something a bundler decided.

It also means the runtime can be sized to the machine: the WebGPU build is 27 MB and the
plain WASM build 14 MB, and a browser with no GPU adapter never downloads the GPU one.

---

## 2026-10-28 — Model hosting

No Hugging Face token is available in this environment, so the 89 MB `model.onnx` ships as
a **GitHub Release asset** rather than being committed (`.gitignore` excludes `*.onnx`).
The worker fetches it with a streaming reader for progress, then stores it in the **Cache
API** so a second visit starts instantly. `VITE_MODEL_URL` overrides the location, which is
how you would point the app at a Hugging Face repo, R2 bucket or your own server instead.

**Correction, after the first deploy.** The plan was for the browser to fetch the release
asset directly. It cannot: GitHub serves release assets with **no
`access-control-allow-origin` header at all**, so a `fetch()` from the Pages origin is
blocked by CORS. Verified rather than assumed, after the fact:

```
$ curl -sIL -H 'Origin: https://jaworskig00.github.io' <asset url> | grep -i access-control
(nothing)
```

The fix keeps every property the release-asset plan had: the deploy workflow downloads the
asset into `dist/model/` before publishing, so the browser fetches the weights **from the
app's own origin** and CORS never enters into it. The blob still never touches git history,
and the release stays the canonical copy. Hosting elsewhere via `VITE_MODEL_URL` still works
— the requirement is simply that the host sends CORS headers.

---

## 2026-10-28 — Canvas for the head grid, SVG for everything else

6 layers x 12 heads of a `T x T` causal matrix is 294 912 cells at a 64-token context.
As SVG `<rect>` elements that is a third of a million DOM nodes per repaint, which no
browser survives at interactive rates. The grid is therefore one canvas: a `T x T`
`ImageData` is written once per head into a reusable offscreen canvas and scaled up with
`drawImage` and smoothing off, so a full repaint is 72 draw calls, around 3 ms. Viridis is
baked into a 256-entry byte lookup table, because calling d3's interpolator per pixel means
300 000 string allocations and parses per frame.

Everything else stays SVG. The arc diagram is a couple of dozen paths after thresholding and
needs hit testing, tooltips and a tweened `d` attribute; the Sankey and the tree are tens of
elements with transitions. D3 owns those elements end to end through keyed joins — React
owns the layout and never touches their internals.

---

## 2026-10-28 — The Sankey is a transport plan, not a bump chart

The logit lens produces a _distribution per layer_, and d3-sankey wants a _conserved flow_.
Three constructions were tried on paper; two of them lie.

Linking `(layer, token) -> (layer+1, token)` with the next layer's probability makes node
inflow and outflow disagree, and d3-sankey papers over it with `max(in, out)`, so the ribbon
widths stop meaning anything. Adding an "other" bucket to absorb the difference does not
close either, because a layer's candidates can gain more mass than the bucket holds.

What does close: renormalise over the selected candidates so every layer carries exactly
1.0, then treat each layer transition as transport. A candidate keeps `min(p, p_next)` and
sheds the remainder; since both sides sum to 1, total surplus equals total deficit exactly,
so the shed mass can be matched to the gainers with nothing left over. The first version
used a proportional (surplus x deficit) split, which is correct but draws up to 16 ribbons
per transition and reads as a barcode. The **north-west corner rule** gives the same
marginals with at most `|surplus| + |deficit| - 1` ribbons. Conservation is asserted per
layer and per node in `tests/lens.test.ts`.

---

## 2026-10-28 — Things that only showed up once the real model was running

Written down because each one was invisible in review and obvious in the browser:

- **The head grid rendered black.** d3's `interpolateViridis` returns `#440154`, not
  `rgb(68, 1, 84)`, so the regex building the colour LUT never matched and every entry
  stayed zero. Parsed with `d3.rgb` now, which handles whatever a scheme returns.
- **d3-sankey parks isolated zero-value nodes in the first column** as a stack of slivers,
  which also squeezed the real ribbons into the top third of the chart. Candidates with no
  mass at a layer simply get no node there.
- **The residual-stream strip was flat black.** Norms run from 4.9 at the embedding to 1810
  at the output, so a linear ramp puts everything but the last row at the bottom of the
  scale. Logarithmic now, and the caption says so.
- **`int32` at the graph boundary.** ORT's WebGPU execution provider has patchy `int64`
  support, so the export casts the input and the logit-lens indices; the only `int64` left
  is what GPT-2 itself needs internally.

---

## 2026-10-28 — WebGPU is the deployed default, and it is not the fastest option

Measured on an Apple M4 Pro in Chromium 152, prompt + 16 tokens at a 25-token context:

| backend         | start-up (cached)            | throughput | per pass |
| --------------- | ---------------------------- | ---------- | -------- |
| WebGPU          | 0.75 s                       | 17.9 tok/s | 57 ms    |
| WASM, 4 threads | 0.85 s                       | 27.3 tok/s | 37 ms    |
| WASM, 1 thread  | > 2 min to build the session | —          | —        |

Threaded WASM wins, which is not the intuitive result: DistilGPT-2 at this context length is
small enough that per-dispatch overhead and the CPU fallback for ops like `TopK` dominate,
and Apple Silicon's SIMD path is strong. But threads need `SharedArrayBuffer`, which needs
COOP/COEP response headers, which **GitHub Pages cannot send**. Single-threaded, session
creation never finished inside two minutes of waiting — not a slow path, an unusable one.

So: WebGPU is the primary path on the deployed build; `public/_headers` ships the isolation
headers for hosts that honour them (Cloudflare Pages, Netlify), where WASM becomes the fast
path; and a browser with no WebGPU adapter is **told** in the loading screen and offered the
recorded run rather than left watching a progress bar that will not move. `?backend=wasm`
pins the CPU path so the fallback can be tried deliberately.

The fix not taken: a COOP/COEP service worker (the `coi-serviceworker` pattern) would give
cross-origin isolation on a static host. It was left out because `credentialless` is not
supported in Safari, and getting it wrong would break the cross-origin model download for
the browsers that currently work. Noted as the obvious next step.

---

## 2026-10-28 — The recorded run is a feature, not just a test fixture

CI must not download 89 MB, so the e2e suite needed a mocked worker. Rather than a mock that
only Playwright can see, `scripts/record_fixture.mjs` records a real run of the ONNX graph —
every step, every attention matrix, 31 tree expansions — into `public/demo-run.json`, and a
stand-in worker replays it through the **same protocol** the real worker speaks.

That one artefact does three jobs: CI runs offline in six seconds, `?demo=1` gives phones
and metered connections the entire interface on genuine model output, and the README media
is captured from it. It also keeps the protocol honest — if the UI works against the replay,
the worker boundary really is the only coupling.

Attention is quantised to 8 bits before base64 encoding, which is the difference between a
329 kB fixture and a multi-megabyte one, and is far more precision than a heatmap or an arc
width can carry.

---

## Final state

Everything in the brief is built and verified by running it, not by assuming:

- DistilGPT-2 exported with attention, hidden states, teacher-forced token log-probabilities
  and a per-layer logit lens, quantised to **89 MB**, verified before any UI existed.
- All six views, linked to one selected step and token, with keyboard playback (`←` `→`,
  space), ARIA labels, a colour-vision-safe palette and reduced-motion support.
- **68 unit tests** over the sampling maths, attention indexing and normalisation, the
  Sankey transport plan and the tree reducer; **8 Playwright tests** against the recorded
  run.
- CI runs lint, format, typecheck, unit tests, e2e and build. Deploy publishes to GitHub
  Pages and then fetches the live URL, failing if it does not serve the app.

### Left undone

- **Nothing is published yet.** Creating the public repository was blocked in the
  environment this was built in, so the GitHub repo, the first push, the Release asset
  (`./scripts/publish_model.sh`) and the Pages deployment still need to be run by hand. The
  CI and deploy workflows are written and the deploy verifies the live URL itself, but
  neither has executed.
- The demo links in this README and the release URL in `src/worker/client.ts` assume
  `jaworskig00/inside-the-llm`. They go live with the first deploy.
- The COOP/COEP service worker described above, which would make the WASM fallback usable on
  GitHub Pages.
- Forking from a tree branch re-runs generation with the forked prefix treated as prompt, so
  those tokens lose their "generated" styling in the token strip. Correct output, slightly
  lossy provenance.

### Next ideas

- A second model to switch between (GPT-2 small is 6x the parameters and 12 layers, which
  would make the layer-flow view considerably more interesting).
- Head labelling: previous-token, induction and attention-sink heads are visually obvious in
  the grid and could be detected and named automatically.
- Attention patching — ablate a head and show, live, which tokens change.
