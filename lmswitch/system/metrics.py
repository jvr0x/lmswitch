"""Token-counter scraping from a live model server.

Every runtime lmswitch drives exposes cumulative prompt/generation token
counters on ``/metrics`` in Prometheus text format, under a runtime-specific
metric name.  The counters live inside the server process and die with it, so
they have to be read *before* the server is stopped — see
``lmswitch.system.usage.sample_server``.

Counter names are verified against primary sources, not guessed:

    ``llama``   ``llama.cpp/tools/server/server-context.cpp`` — emits every
                metric under a ``llamacpp:`` prefix, and serves ``/metrics``
                only when the server was started with ``--metrics``.
    ``vllm``    read off a live serving container's ``/metrics``.
    ``sglang``  ``sglang/srt/observability/metrics_collector.py`` — needs
                ``--enable-metrics``.

Both flags are added automatically at launch (``LlamaRuntime._build_cmd``,
``_sglang_args``), so a recipe needs no changes to be accounted for.
"""

from __future__ import annotations

import urllib.request

# (prompt counter, generation counter) per runtime family.
#
# Reason: kept closed on purpose. A name lands here only after being read off
# a live server, and the summing in `_counter_value` is valid only for counters
# whose labels *partition* the total (vLLM's `engine` / `model_name`), never
# for ones that re-slice it — `vllm:prompt_tokens_by_source_total` carries the
# same tokens again split by `source`, and adding it would double-count.
_COUNTERS: dict[str, tuple[str, str]] = {
    "llama": ("llamacpp:prompt_tokens_total", "llamacpp:tokens_predicted_total"),
    "vllm": ("vllm:prompt_tokens_total", "vllm:generation_tokens_total"),
    "sglang": ("sglang:prompt_tokens_total", "sglang:generation_tokens_total"),
}


def _family(runtime: str) -> str:
    """Returns the metrics family for *runtime*.

    The ``-dual`` / ``-dual-ray`` variants serve the same engine behind the
    same API server on the head node, so they report under the base runtime's
    metric names.

    Args:
        runtime: Runtime type string (e.g. ``"vllm-dual-ray"``).

    Returns:
        The base family name (e.g. ``"vllm"``).
    """
    return runtime.split("-", 1)[0]


def _counter_value(text: str, name: str) -> int:
    """Sums every sample of the Prometheus counter *name* in *text*.

    Args:
        text: Raw ``/metrics`` response body.
        name: Fully qualified metric name, prefix included.

    Returns:
        The summed counter value, or 0 when the metric is absent.
    """
    total = 0.0
    for line in text.splitlines():
        line = line.strip()
        if not line or line.startswith("#") or not line.startswith(name):
            continue
        rest = line[len(name):]
        # Reason: a sample name must end here — the next character is either a
        # label set or the whitespace before the value. Without this check a
        # metric would swallow any longer name it happens to prefix.
        if rest[:1] not in ("{", " ", "\t"):
            continue
        if rest.startswith("{"):
            close = rest.rfind("}")
            if close < 0:
                continue
            rest = rest[close + 1:]
        parts = rest.split()
        if not parts:
            continue
        try:
            # Prometheus encodes counters as floats ("3.675428e+06"), so int()
            # on the raw token would raise.
            total += float(parts[0])
        except ValueError:
            continue
    return int(total)


def fetch_token_counters(port: int, runtime: str, timeout: float = 3.0) -> tuple[int, int]:
    """Reads cumulative token counters off a running server's ``/metrics``.

    Never raises and never blocks longer than *timeout*: a server that is
    already down, was built without its metrics endpoint, or answers garbage
    just accounts as zero rather than breaking the stop path.

    Args:
        port: Port the server listens on. 0 skips the scrape.
        runtime: Runtime type string, used to pick the counter names.
        timeout: Socket timeout in seconds.

    Returns:
        ``(prompt_tokens, generation_tokens)`` — cumulative over the life of
        the server process, so they reset whenever it restarts.
    """
    names = _COUNTERS.get(_family(runtime))
    if port <= 0 or names is None:
        return 0, 0
    try:
        url = f"http://127.0.0.1:{port}/metrics"
        with urllib.request.urlopen(url, timeout=timeout) as resp:
            text = resp.read().decode("utf-8", errors="replace")
    except Exception:
        return 0, 0
    return _counter_value(text, names[0]), _counter_value(text, names[1])
