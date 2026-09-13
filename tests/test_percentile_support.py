# Copyright (c) 2024 AXELTEC SOFTWARE LTD.
# SPDX-License-Identifier: Apache-2.0

"""A percentile must be backed by enough requests to mean something.

Levels run for seconds, not minutes, so a sweep collects tens to low hundreds
of requests per point. At those counts p99 rests on one or two observations and
p99.9 is the maximum. Reported without their sample size they read as tail
measurements, and a reader has no way to know they are not. This tool exists to
report distributions rather than averages; publishing a percentile it cannot
support is the same failure in the other direction.
"""

import json

from metron.__main__ import print_summary, supported_percentiles


def test_a_short_run_cannot_carry_the_tail():
    ok = supported_percentiles(206)  # a 15s level at ~14 req/s
    assert ok["p(90)"] is True
    assert ok["p(99)"] is False
    assert ok["p(99.9)"] is False


def test_a_long_run_can():
    ok = supported_percentiles(12000)
    assert all(ok.values())


def test_the_boundary_is_ten_observations_past_the_percentile():
    assert supported_percentiles(99)["p(90)"] is False
    assert supported_percentiles(100)["p(90)"] is True
    assert supported_percentiles(999)["p(99)"] is False
    assert supported_percentiles(1000)["p(99)"] is True


def _summary(tmp_path, n):
    d = {
        "metrics": {
            "http_reqs": {"count": n},
            "ttft": {"med": 27.0, "p(90)": 44.0, "p(99)": 547.0, "max": 632.0},
        }
    }
    p = tmp_path / "summary.json"
    p.write_text(json.dumps(d))
    return str(p)


def test_the_sample_count_is_printed(tmp_path, capsys):
    print_summary(_summary(tmp_path, 206))
    assert "n=206" in capsys.readouterr().out


def test_an_unsupported_percentile_is_withheld_not_guessed(tmp_path, capsys):
    print_summary(_summary(tmp_path, 206))
    out = capsys.readouterr().out
    assert "547" not in out, "p99 printed from 206 requests"
    assert "—" in out
    assert "too few" in out


def test_a_supported_percentile_is_shown(tmp_path, capsys):
    print_summary(_summary(tmp_path, 5000))
    out = capsys.readouterr().out
    assert "547" in out and "44" in out


def test_the_k6_summary_no_longer_requests_p999():
    import pathlib

    src = (
        pathlib.Path(__file__).resolve().parents[1] / "metron" / "__main__.py"
    ).read_text()
    assert "p(99.9)" not in src.split("summaryTrendStats")[-1][:200]
