# Copyright (c) 2024 AXELTEC SOFTWARE LTD.
# SPDX-License-Identifier: Apache-2.0

"""One bad point must not abort a whole sweep.

Both cases here are real, caught by running modern instruct models through the
raw completions route: a one-token completion made TPOT divide by zero (+Inf,
which k6 cannot marshal), and a present-but-null percentile crashed the summary
printer formatting None."""

from pathlib import Path

from metron.metrics import run_health


def test_print_summary_tolerates_null_percentiles(tmp_path, capsys):
    import json
    from metron.__main__ import print_summary

    # ttft present but its median is null — the shape a degenerate run produces.
    summary = {
        "metrics": {
            "ttft": {"med": None, "p(90)": 5.0, "p(99)": 9.0, "max": 9.0},
            "itl": {"med": 6.0, "p(90)": 6.0, "p(99)": 7.0, "max": 8.0},
        }
    }
    p = tmp_path / "summary.json"
    p.write_text(json.dumps(summary))
    print_summary(str(p))  # must not raise
    out = capsys.readouterr().out
    assert "TTFT" in out and "nan" in out.lower()


def test_tpot_guard_is_present_in_the_k6_script():
    """The generator must not compute TPOT for a one-token completion; that
    divisor is zero and produced the +Inf that broke JSON marshalling."""
    js = (Path(__file__).resolve().parents[1] / "metron" / "metron.js").read_text()
    assert "total_tokens > 1" in js, "TPOT divide-by-zero guard missing"


def test_run_health_still_flags_the_failures_behind_a_bad_point():
    # The other half of honesty: the failures that caused the degenerate point
    # are surfaced, not swallowed.
    s = {
        "metrics": {"dropped_iterations": {"count": 3}},
        "root_group": {"checks": {"is status 200": {"passes": 90, "fails": 10}}},
    }
    assert len(run_health(s)) == 2


def test_missing_token_counter_does_not_raise():
    """A run where no request produced countable output has no token counter at
    all: k6 omits a Counter that was never incremented. Postprocessing must
    report zero rather than raise KeyError."""
    from metron.metrics import calculate_common_throughput, get_total_duration

    summary = {"metrics": {}, "root_group": {"checks": {}}}
    assert get_total_duration(summary) == 0.0
    assert calculate_common_throughput(summary) == 0.0
