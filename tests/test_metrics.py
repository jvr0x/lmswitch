"""Tests for token-counter scraping (``lmswitch.system.metrics``).

The parser is exercised against a ``/metrics`` body captured verbatim from a
live vLLM server (``tests/fixtures/vllm_metrics.txt``) rather than only
synthesized text, so it is tested against the real wire format — scientific
notation, label sets and all.
"""

from pathlib import Path
from unittest import mock

import pytest

from lmswitch.system import metrics as metrics_mod
from lmswitch.system.metrics import _counter_value, _family, fetch_token_counters

FIXTURE = Path(__file__).parent / "fixtures" / "vllm_metrics.txt"

# Captured from the live server the fixture came from.
LIVE_PROMPT_TOKENS = 3675428
LIVE_GENERATION_TOKENS = 44995


@pytest.fixture
def vllm_body() -> str:
    """Raw /metrics body from a real vLLM server."""
    return FIXTURE.read_text()


# ---------------------------------------------------------------------------
# Expected use
# ---------------------------------------------------------------------------

def test_parses_real_vllm_body(vllm_body):
    """Both counters come back exactly as the live server reported them."""
    assert _counter_value(vllm_body, "vllm:prompt_tokens_total") == LIVE_PROMPT_TOKENS
    assert _counter_value(vllm_body, "vllm:generation_tokens_total") == LIVE_GENERATION_TOKENS


def test_fetch_returns_both_counters(vllm_body):
    """fetch_token_counters pairs the runtime's two counters."""
    with mock.patch.object(metrics_mod, "urllib") as mock_urllib:
        mock_urllib.request.urlopen.return_value.__enter__.return_value.read.return_value = \
            vllm_body.encode()
        assert fetch_token_counters(8888, "vllm") == (LIVE_PROMPT_TOKENS,
                                                      LIVE_GENERATION_TOKENS)


@pytest.mark.parametrize("prefix,gen_name", [
    ("llamacpp", "tokens_predicted_total"),
    ("sglang", "generation_tokens_total"),
])
def test_llama_and_sglang_counter_names(prefix, gen_name):
    """Each runtime's own metric names are the ones read."""
    runtime = "llama" if prefix == "llamacpp" else "sglang"
    body = (f"# TYPE {prefix}:prompt_tokens_total counter\n"
            f"{prefix}:prompt_tokens_total 1200\n"
            f"{prefix}:{gen_name} 340\n")
    with mock.patch.object(metrics_mod, "urllib") as mock_urllib:
        mock_urllib.request.urlopen.return_value.__enter__.return_value.read.return_value = \
            body.encode()
        assert fetch_token_counters(8081, runtime) == (1200, 340)


@pytest.mark.parametrize("runtime,family", [
    ("llama", "llama"), ("llama-dual", "llama"),
    ("vllm", "vllm"), ("vllm-dual", "vllm"), ("vllm-dual-ray", "vllm"),
    ("sglang", "sglang"), ("sglang-dual", "sglang"),
])
def test_dual_variants_share_the_base_family(runtime, family):
    """Dual runtimes report under the base runtime's metric names."""
    assert _family(runtime) == family


# ---------------------------------------------------------------------------
# Edge cases
# ---------------------------------------------------------------------------

def test_sums_across_label_sets():
    """A counter split over several engines adds up to the total."""
    body = ('vllm:generation_tokens_total{engine="0",model_name="m"} 100.0\n'
            'vllm:generation_tokens_total{engine="1",model_name="m"} 250.0\n')
    assert _counter_value(body, "vllm:generation_tokens_total") == 350


def test_parses_scientific_notation():
    """Prometheus float encoding is the normal case for large counters."""
    body = 'vllm:prompt_tokens_total{engine="0"} 3.675428e+06\n'
    assert _counter_value(body, "vllm:prompt_tokens_total") == 3675428


def test_name_match_stops_at_the_sample_boundary(vllm_body):
    """A counter never swallows a longer metric it happens to prefix.

    ``vllm:prompt_tokens_by_source_total`` re-slices the very same tokens by
    source, so counting it too would roughly double the recorded prompt total.
    """
    by_source = _counter_value(vllm_body, "vllm:prompt_tokens_by_source_total")
    # The re-slice sums to the same underlying total — which is exactly why
    # the two must never be read as one metric.
    assert by_source == LIVE_PROMPT_TOKENS
    assert _counter_value(vllm_body, "vllm:prompt_tokens_total") == LIVE_PROMPT_TOKENS

    sliced_only = 'vllm:prompt_tokens_by_source_total{source="local_compute"} 999\n'
    assert _counter_value(sliced_only, "vllm:prompt_tokens_total") == 0


def test_ignores_help_and_type_lines():
    """Comment lines naming the metric contribute nothing."""
    body = ("# HELP vllm:prompt_tokens_total Number of prefill tokens.\n"
            "# TYPE vllm:prompt_tokens_total counter\n"
            "vllm:prompt_tokens_total 42\n")
    assert _counter_value(body, "vllm:prompt_tokens_total") == 42


def test_unlabelled_sample_is_read():
    """llama.cpp emits bare ``name value`` lines with no label set."""
    assert _counter_value("llamacpp:prompt_tokens_total 7\n",
                          "llamacpp:prompt_tokens_total") == 7


# ---------------------------------------------------------------------------
# Failure cases
# ---------------------------------------------------------------------------

def test_absent_metric_is_zero(vllm_body):
    """A vLLM body holds no llama.cpp counters."""
    assert _counter_value(vllm_body, "llamacpp:prompt_tokens_total") == 0


def test_port_zero_never_opens_a_connection():
    """No port means no scrape at all, not a failed one."""
    with mock.patch.object(metrics_mod, "urllib") as mock_urllib:
        assert fetch_token_counters(0, "vllm") == (0, 0)
    mock_urllib.request.urlopen.assert_not_called()


def test_unknown_runtime_never_opens_a_connection():
    """An unrecognised runtime has no counter names to ask for."""
    with mock.patch.object(metrics_mod, "urllib") as mock_urllib:
        assert fetch_token_counters(8080, "some-future-runtime") == (0, 0)
    mock_urllib.request.urlopen.assert_not_called()


def test_unreachable_server_is_zero_not_an_exception():
    """A server that is already down must never break the stop path."""
    with mock.patch.object(metrics_mod.urllib.request, "urlopen",
                           side_effect=OSError("connection refused")):
        assert fetch_token_counters(8888, "vllm") == (0, 0)


def test_garbage_body_is_zero():
    """A non-Prometheus response (404 HTML, JSON error) accounts as zero."""
    body = '{"error":{"message":"This server does not support metrics"}}'
    with mock.patch.object(metrics_mod, "urllib") as mock_urllib:
        mock_urllib.request.urlopen.return_value.__enter__.return_value.read.return_value = \
            body.encode()
        assert fetch_token_counters(8081, "llama") == (0, 0)


def test_unparsable_value_is_skipped():
    """A malformed sample is dropped, the well-formed ones still count."""
    body = ("vllm:prompt_tokens_total{engine=\"0\"} not-a-number\n"
            "vllm:prompt_tokens_total{engine=\"1\"} 5\n")
    assert _counter_value(body, "vllm:prompt_tokens_total") == 5
