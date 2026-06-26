---
name: lmswitch-recipes
description: Author lmswitch model recipes (ai-models/*.yaml) for serving local LLMs via llama-server (GGUF) or vLLM (Docker). Use when adding a new model to lmswitch, editing a recipe, or passing custom serving flags.
---

# Writing lmswitch recipes

lmswitch serves local LLMs from one YAML file per model in `ai-models/`. The
**filename minus `.yaml` is the model id** — it's also the `--alias` /
`--served-model-name` and (for vLLM) the container name `vllm-<id>`. Drop a file
in, run `lmswitch on <id>`. No code changes needed.

Two runtimes:
- `runtime: llama` (default) → GGUF via `llama-server`, background process.
- `runtime: vllm` → safetensors/quantized via `vllm-openai` Docker container.

`model:` is **always a path relative to `MODELS_DIR`** (default `~/models`,
override in `ai-models/.lmswitch`). For GGUF it points at the `.gguf` file (or
the `-00001-of-000NN.gguf` first shard of a split); for vLLM it points at the
model **directory**.

## Minimal recipes

GGUF (`ai-models/my-model.yaml`):
```yaml
runtime: llama
model: "unsloth/Qwen3-4B-GGUF/Qwen3-4B-Q4_K_M.gguf"
port: 8085
ctx: 65536
display_name: "Qwen3-4B"
```

vLLM:
```yaml
runtime: vllm
model: "nvidia/qwen3.6-35b-a3b-nvfp4"
port: 8114
ctx: 32768
display_name: "Qwen3.6-35B NVFP4"
gpu_memory_utilization: 0.55
```

## llama (GGUF) keys

| Key | Default | Maps to / notes |
|---|---|---|
| `model` | — (required) | `--model`, relative to MODELS_DIR |
| `port` | — (required) | server port, must be unique |
| `ctx` | 65536 | `--ctx-size` |
| `display_name` | = id | table/opencode label |
| `gpu_layers` | 99 | `--n-gpu-layers` (99 = offload all) |
| `threads` | 12 | `--threads` |
| `batch` / `ubatch` | 1024 / 512 | `--batch-size` / `--ubatch-size` |
| `alias` | = id | `--alias` |
| `mmproj` | — | multimodal projector path (rel. MODELS_DIR) for vision GGUFs |
| `fit` | `off` | `-fit`. **Keep `off` on GB10/Blackwell** (auto-fit's `cudaMemGetInfo` aborts). Use `none` to omit the flag entirely on old llama.cpp builds |
| `llama_bin` | `../llama.cpp/build/bin/llama-server` | override the binary |
| `ready_timeout` | 300 | seconds to wait for port to bind |
| `force` | false | bypass the pre-load RAM guard |
| `restart` | — | `on-failure` opts into systemd auto-restart |
| `extra_args` | — | **any other llama-server flag** (see below) |

## vLLM keys

| Key | Default | Maps to / notes |
|---|---|---|
| `model` | — (required) | model **dir**, relative to MODELS_DIR |
| `port` | — (required) | server port, must be unique |
| `ctx` | 65536 | `--max-model-len` |
| `display_name` | = id | label |
| `gpu_memory_utilization` | 0.15 | `--gpu-memory-utilization`. Low (~0.15) to coexist with a running llama-server; high (~0.85) for a big model running alone |
| `image` | `vllm/vllm-openai:cu130-nightly` | Docker image |
| `enforce_eager` | true | emits `--enforce-eager` **by default**; set `false` to allow CUDA graphs |
| `tool_call_parser` | — | `--enable-auto-tool-choice --tool-call-parser <v>` |
| `reasoning_parser` | — | `--reasoning-parser=<v>` |
| `reasoning_parser_plugin` | — | `--reasoning-parser-plugin=<v>` |
| `trust_remote_code` | — | `--trust-remote-code` (new/custom archs) |
| `limit_mm_per_prompt` | — | `--limit-mm-per-prompt=<v>` |
| `chat_template` | — | `--chat-template=<v>` (skip with `no_chat_template: true`) |
| `attention_backend` | — | `--attention-backend=<v>` |
| `max_num_seqs` | 32 | `--max-num-seqs` |
| `load_format` | — | `--load-format=<v>` |
| `ready_timeout` | 600 | readiness wait |
| `force` / `restart` | — | as above |
| `extra_args` | — | **any other `vllm serve` flag** |

## extra_args — pass ANY flag

Works for both runtimes; appended **last** so it overrides earlier defaults.
Accepts a YAML list (one argv token per item) **or** a single shell-split string:
```yaml
extra_args: ["-fa", "on", "-ctk", "q8_0", "-ctv", "q8_0"]      # llama
extra_args: "--temp 0.7 --top-p 0.9 --jinja"                    # llama
extra_args: ["--enable-prefix-caching", "--kv-cache-dtype", "fp8"]  # vllm
```
Discover flags with `llama-server --help` or `vllm serve --help`.

**vLLM caveat:** these flags are already hardcoded in the docker command —
`--port`, `--host`, `--max-model-len`, `--tensor-parallel-size`,
`--served-model-name`, plus `--gpu-memory-utilization`/`--max-num-seqs` (from
their keys). Don't repeat them in `extra_args` (vLLM would see them twice). Use
the first-class keys. extra_args run **inside the container**, so any file path
must be mounted in (the model dir and `~/.cache/huggingface` are).

## Conventions & gotchas

- **Unique ports.** Check the table first: `lmswitch list`. (Current pool starts at 8081.)
- **Section grouping** is by a substring of the **id**, matched against:
  `qwen`/`qwopus` → Qwen, `gemma`/`diffusiongemma` → Gemma, `deepseek`,
  `nemotron`, `kimi`, `gpt`/`gpt-oss`, `glm`, `nex`, `step`, `ornith`.
  An id matching none lands under **"Other"** — to add a new family, append a
  rule to `FAMILY_RULES` in the `lmswitch` script.
- **GB10 GGUF optimization:** add `extra_args: ["-fa", "on", "-ctk", "q8_0", "-ctv", "q8_0"]`
  — flash-attention plus q8_0 KV cache, which slashes KV memory at long ctx.
  (Quantized KV **requires** flash-attn on, so set both together.)
- **Big vLLM models can't coexist** with a large llama-server on one GB10 — a
  65GB bf16 model at `gpu_memory_utilization: 0.85` needs the GPU to itself.
- **Vision GGUFs** need both the model `.gguf` and an `mmproj:` projector.

## Workflow to add a model

1. Confirm the weights exist under MODELS_DIR (`ls ~/models/<path>`).
2. Pick a free `port` (`lmswitch list`).
3. Pick `runtime` by format: `.gguf` → llama; safetensors/NVFP4/FP8 dir → vllm.
4. Write `ai-models/<id>.yaml` (copy `examples/llama-gguf.yaml` or `examples/vllm.yaml`).
5. Add optimizations via `extra_args` (see GB10 note).
6. Verify it registers: `lmswitch list` (look for `✓` downloaded, right port).
7. Start: `lmswitch on <id>` → wait for "Ready on port N". On failure, read the
   log path it prints (GGUF: `ai-models/running/<id>.log`; vLLM: `docker logs vllm-<id>`).

See the fully-commented templates: `examples/llama-gguf.yaml` (text GGUF),
`examples/llama-gguf-vision.yaml` (vision GGUF with `mmproj`), and
`examples/vllm.yaml`.
