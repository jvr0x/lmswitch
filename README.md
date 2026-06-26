```
██╗     ███╗   ███╗███████╗██╗    ██╗██╗████████╗ ██████╗██╗  ██╗
██║     ████╗ ████║██╔════╝██║    ██║██║╚══██╔══╝██╔════╝██║  ██║
██║     ██╔████╔██║███████╗██║ █╗ ██║██║   ██║   ██║     ███████║
██║     ██║╚██╔╝██║╚════██║██║███╗██║██║   ██║   ██║     ██╔══██║
███████╗██║ ╚═╝ ██║███████║╚███╔███╔╝██║   ██║   ╚██████╗██║  ██║
╚══════╝╚═╝     ╚═╝╚══════╝ ╚══╝╚══╝ ╚═╝   ╚═╝    ╚═════╝╚═╝  ╚═╝
```

# lmswitch

> local LLM switcher · GGUF + vLLM

List and toggle local LLMs from per-model YAML configs.

> **Adding a model?** See [`SKILL.md`](SKILL.md) — a self-contained guide (for
> humans or AI agents) to authoring `ai-models/*.yaml` recipes: every key, the
> `extra_args` passthrough, and the GB10 gotchas. Point an agent at this folder
> and it can add a model from that doc alone.

`lmswitch` shows a table of every configured model (grouped by family) with its
size, download state, port, and whether it's currently serving — and lets you
start/stop them interactively. **GGUF** models run under
[`llama.cpp`](https://github.com/ggml-org/llama.cpp) (`llama-server` as a
background process); **vLLM** models run in Docker. It waits for each model to
actually become ready, refuses loads that would exceed free RAM, and keeps your
coding agents' configs in sync — [opencode](https://opencode.ai),
[hermes](https://github.com/NousResearch/hermes-agent), and
[grok](https://github.com/xai-org/grok-cli) — with whatever is serving.

Running `lmswitch` (the same wordmark above greets you):

```
  RAM    │ 122Gi total   33Gi used   89Gi available
  Models │ ~20.8G weights   1 / 3 loaded
  Disk   │ 113.1G   3 / 3 downloaded
  ● loaded   ○ stopped      ✓ downloaded   ✗ missing

   #  S  TYPE   NAME            SIZE  DL   PORT  DISPLAY
   Qwen ──────────────────────────────────────────────────────────────────────
   1  ○   gguf   qwen3-4b        2.3G  ✓   8085  Qwen3-4B
   2  ●   gguf   qwen3.6-35b    20.8G  ✓   8089  Qwen3.6-35B-A3B
   Nex ───────────────────────────────────────────────────────────────────────
   3  ○   gguf   nex-n2-pro     90.0G  ✓   8104  Nex-N2-Pro 397B-A17B (IQ1_M)

  Toggle # (space/comma separated, enter or q to quit):
```

## Features

- **One table for everything** — loaded (`●`) vs stopped (`○`), downloaded (`✓`)
  vs missing (`✗`), per-model size/port, and RAM / disk / loaded-count totals.
- **Two runtimes** — GGUF via `llama-server`, safetensors/quantized via vLLM in
  Docker. Pick per model with `runtime:`.
- **Readiness-aware** — after launch it polls the model's `/v1/models` endpoint
  and only reports `Ready` once it's actually serving (with a `…loading` progress
  heartbeat and crash detection), so the synced configs reflect reality, not guesses.
- **Pre-load RAM guard** — refuses a start that would blow past available memory
  (overridable per-model with `force: true`), so a too-big model can't OOM-lock
  the machine.
- **Config sync** — on every toggle / `on` / `off` / `sync`, the
  currently-serving models are written into your coding agents' configs:
  **opencode** (`opencode.json`), **hermes** (`config.yaml`), and **grok**
  (`config.toml`). Pick which targets are active during `lmswitch init`. See
  [Config sync](#config-sync).
- **Optional systemd auto-restart** per model via `restart: on-failure`.

## Requirements

lmswitch targets **Linux** (it uses `/proc/meminfo`, `ss`, Docker `--gpus`, and
systemd user units).

| For | You need |
|-----|----------|
| lmswitch itself | Python 3.9+, `curl`, `ss` (iproute2). `pyyaml` is optional — a minimal built-in parser is used if it's absent. |
| GGUF models | A built `llama.cpp` with `llama-server` (a CUDA build for GPU offload). Default binary path: `<lmswitch>/../llama.cpp/build/bin/llama-server` — override per-model with `llama_bin:`. |
| vLLM models | Docker + the [NVIDIA Container Toolkit](https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/latest/) (`--gpus all`). Pulls the `vllm/vllm-openai` image. |
| `restart: on-failure` | A running systemd **user** instance (`systemctl --user`). |
| config sync | Any of [opencode](https://opencode.ai), [hermes](https://github.com/NousResearch/hermes-agent), [grok](https://github.com/xai-org/grok-cli) (all optional — only configs that exist are synced). |

## Install

```bash
# from the repo dir (e.g. ~/utils/lmswitch)
./lmswitch init
```

`init` asks where your models live (writes `ai-models/.lmswitch`), creates the
`ai-models/` config dir, symlinks `~/.local/bin/lmswitch` → the script, and asks
which sync targets to enable (opencode / hermes / grok — only the ones whose
configs it finds). Ensure `~/.local/bin` is on your `$PATH`. (Manual alternative:
`ln -sf "$PWD/lmswitch" ~/.local/bin/lmswitch`.)

## Getting started

1. Install the requirements above.
2. `lmswitch init` — set your models directory (default `~/models`).
3. Download a model into that directory (see **Where to get models**).
4. Create a config — `lmswitch add <name>`, or copy a template from
   [`examples/`](examples/) into `ai-models/<name>.yaml`.
5. `lmswitch` → type the model's number to start it. It loads, waits until the
   endpoint answers, prints `Ready on port <port>`, and syncs your enabled
   configs (opencode / hermes / grok).
6. Hit it: `curl localhost:<port>/v1/models`.

## Usage

```
lmswitch                  # interactive: show the table, then type model #s to toggle
lmswitch list             # just print the table (read-only)
lmswitch on  <name|#>     # start a model
lmswitch off <name|#>     # stop a model
lmswitch sync             # regenerate enabled configs from currently-serving models
lmswitch add  <name>      # create a model config interactively
lmswitch serve <name>     # run a model in the foreground (used by systemd)
lmswitch init             # bootstrap ai-models/, .lmswitch, and the symlink
```

In the interactive prompt you can toggle several at once, space/comma separated:
`8 9 24`. A toggle blocks until the model is ready (or its `ready_timeout`
elapses); `Ctrl-C` aborts cleanly — the model keeps loading detached, so re-run
`lmswitch` or `lmswitch sync` to pick it up once it's up.

## Model configs (`ai-models/<name>.yaml`)

The filename (minus `.yaml`) is the model's id, its `served-model-name`, and —
for vLLM — its container name (`vllm-<id>`). `model:` is a path **relative to
your models directory**. Fully-commented templates live in
[`examples/llama-gguf.yaml`](examples/llama-gguf.yaml) and
[`examples/vllm.yaml`](examples/vllm.yaml).

**Common keys**

| Key | Default | Meaning |
|-----|---------|---------|
| `runtime` | `llama` | `llama` (GGUF) or `vllm` (Docker) |
| `model` | — | path to the `.gguf` file (llama) or model dir (vLLM), relative to the models dir |
| `port` | `8081` | OpenAI-compatible server port |
| `ctx` | `65536` | context length |
| `display_name` | `<name>` | label in the table / synced configs |
| `ready_timeout` | `600` (vLLM) / `300` (llama) | seconds to wait for readiness |
| `force` | `false` | bypass the pre-load RAM guard |
| `restart` | — | `on-failure` → run under a systemd user unit |

**llama (GGUF) keys**: `gpu_layers` (99), `threads` (12), `batch` (1024),
`ubatch` (512), `alias`, `mmproj`, `llama_bin`, and `fit` — default `off`, which
skips llama.cpp's auto memory-fit step (it aborts in `cudaMemGetInfo` on some
CUDA builds, e.g. GB10/Blackwell); set `fit: none` to omit the flag entirely on
older llama.cpp builds that don't support `-fit`.

**vLLM keys**: `gpu_memory_utilization` (0.15), `image`, `tool_call_parser`,
`reasoning_parser`, `trust_remote_code`, `max_num_seqs`, `extra_args`, and more
— see [`examples/vllm.yaml`](examples/vllm.yaml) for the full list.

## Where to get models

Models come from [Hugging Face](https://huggingface.co) into your models
directory; each config's `model:` path is relative to it.

```bash
pip install -U "huggingface_hub[cli]"

# GGUF (llama.cpp) — e.g. Unsloth / bartowski quants; grab the .gguf file(s)
hf download unsloth/Qwen3-4B-GGUF Qwen3-4B-Q4_K_M.gguf \
  --local-dir ~/models/unsloth/Qwen3-4B-GGUF
#   → model: "unsloth/Qwen3-4B-GGUF/Qwen3-4B-Q4_K_M.gguf"

# vLLM (safetensors, incl. FP8 / NVFP4 quants) — grab the whole repo into a dir
hf download nvidia/Qwen3.5-MoE-...-NVFP4 --local-dir ~/models/nvidia/qwen3-...-nvfp4
#   → model: "nvidia/qwen3-...-nvfp4"
```

Good sources: `unsloth/`, `bartowski/`, `ggml-org/` for GGUF; the upstream model
repos and `nvidia/` (FP8 / NVFP4) for vLLM. Multi-shard GGUF
(`*-00001-of-0000N.gguf`) is detected automatically — point `model:` at the
first shard.

## How it works

- **GGUF** → `llama-server` is launched as a detached background process; its PID
  and full output go to `ai-models/running/<name>` and `…/<name>.log`. `-fit off`
  is passed by default (see `fit:` above).
- **vLLM** → `docker run -d --name vllm-<name> --gpus all --network host …`; any
  stale/exited container of the same name is `docker rm -f`'d first to avoid a
  name conflict.
- **Readiness** → after launch it polls `http://localhost:<port>/v1/models` until
  it answers (`Ready on port <port>`), the process/container dies
  (`✗ … exited during startup` + a pointer to the log / `docker logs`), or
  `ready_timeout` elapses (`WARNING`).
- **RAM guard** → before launching, free RAM (`MemAvailable` from `/proc/meminfo`)
  is compared to an estimate: `gpu_memory_utilization × total` for vLLM, on-disk
  weight size × 1.3 for GGUF. If short, the start is refused unless `force: true`.
- **Config sync** → each enabled target gets the currently-serving models, all
  pointing at `http://<SPARK_HOST>:<port>/v1`: **opencode** one provider per
  model, **hermes** the active model + a `custom_providers` entry per model (so
  they show in `/model`), **grok** one `[model.<id>]` table per model.
  `SPARK_HOST` is a constant near the top of the script (`spark-8912.local`) —
  change it if your host differs. See [Config sync](#config-sync).

## Config sync

lmswitch keeps your coding agents' configs honest: on every `on` / `off` /
toggle / `sync` it rewrites the **currently-serving** models into each enabled
target, every endpoint pointing at `http://<SPARK_HOST>:<port>/v1`. Your agent
always sees the models that are actually up — right ports, right names — with no
hand-editing and no calls to a model that isn't loaded. `SPARK_HOST` is a
constant near the top of the script (`spark-8912.local`); set it to your serving
host.

Targets are chosen during `lmswitch init` and stored as `SYNC_OPENCODE` /
`SYNC_HERMES` / `SYNC_GROK` in `ai-models/.lmswitch` (only configs that exist on
disk are touched; a target with no config is skipped). Each shapes its own file:

- **[opencode](https://opencode.ai)** → `~/.config/opencode/opencode.json` gets
  one provider per serving model. If `~/.local/share/opencode-export/` exists, a
  copy is written there too, so a remote client (a laptop/Mac over Tailscale,
  LAN, or a Samba mount) can pick up the same config and point straight at the
  serving host.
- **[hermes](https://github.com/NousResearch/hermes-agent)** →
  `~/.hermes/config.yaml`. Hermes runs one active model, so the `model:` block is
  set to the serving model and kept **sticky** (only switched when the current
  one stops); a vision model (id containing `vl`) is wired into
  `auxiliary.vision`. Every serving model is also registered under
  `custom_providers:` (with `discover_models: false`, so the picker doesn't
  live-probe and hang) — that's what makes them all selectable from hermes'
  `/model`. Custom providers you added by hand (pointing at other hosts) are
  preserved. Because discovery is off, `/model` reflects the last sync — re-run
  `lmswitch sync` (or just toggle) after starting a model to refresh the list.
- **[grok](https://github.com/xai-org/grok-cli)** → `~/.grok/config.toml` gets
  one `[model.<id>]` table per serving model; all your other grok settings
  (`[cli]`, `[ui]`, marketplace, the `[models] default`, …) are left untouched.

Run `lmswitch sync` to regenerate on demand — handy after a detached load
finishes.

## Development & tests

The tests need `pytest` and `pyyaml`. The easiest setup is a
[uv](https://docs.astral.sh/uv/) virtualenv (the checked-in `.venv` is **not**
portable — if it was copied from another machine/OS, delete and recreate it):

```bash
cd ~/utils/lmswitch
rm -rf .venv                       # only if a stale/foreign .venv is present
uv venv                            # create .venv from pyproject (Python >=3.10)
uv pip install pytest pyyaml       # test deps (pyyaml is also a runtime dep)
```

Run the whole suite (prefix with `uv run` so it uses the venv):

```bash
uv run pytest tests/ -q
```

Or run a single file / test:

```bash
uv run pytest tests/test_sync.py -q
uv run pytest tests/test_sync.py::test_regen_hermes_keeps_running_default_sticky -q
```

| File | Covers |
|------|--------|
| `tests/test_llama_cmd.py` | llama-server command construction |
| `tests/test_vllm_and_abort.py` | vLLM start, readiness, RAM guard, Ctrl-C, opencode sync |
| `tests/test_sync.py` | config sync to opencode / hermes / grok (selection, idempotency, round-trip) |

All tests are pure unit tests — `subprocess` / Docker / `curl` / ports are
stubbed and configs are written to temp dirs, so they run anywhere (no GPU, no
models, no Docker, and they never touch your real configs).

> Without `uv` you can install the deps into any Python 3.10+ environment
> (`pip install pytest pyyaml`) and run `pytest tests/`. The two older files also
> run standalone (`python3 tests/test_llama_cmd.py`); `test_sync.py` needs pytest.

## License

[Apache-2.0](LICENSE) © 2026 jvr0x.
