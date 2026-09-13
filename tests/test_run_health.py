# Copyright (c) 2024 AXELTEC SOFTWARE LTD.
# SPDX-License-Identifier: Apache-2.0

from metron.metrics import run_health


def summary(dropped=0, passes=0, fails=0, empty=0, estimated=0, incomplete=0):
    s = {"metrics": {}, "root_group": {"checks": {}}}
    if dropped:
        s["metrics"]["dropped_iterations"] = {"count": dropped}
    if empty:
        s["metrics"]["responses_without_content"] = {"count": empty}
    if estimated:
        s["metrics"]["token_counts_estimated"] = {"count": estimated}
    if incomplete:
        s["metrics"]["incomplete_streams"] = {"count": incomplete}
    if passes or fails:
        s["root_group"]["checks"]["is status 200"] = {"passes": passes, "fails": fails}
    return s


def test_a_clean_run_reports_nothing():
    assert run_health(summary(passes=100)) == []


def test_dropped_iterations_are_surfaced():
    [msg] = run_health(summary(dropped=12, passes=100))
    # The warning must name the effect, not the queueing-theory term: the
    # reader needs to know the number understates the tail.
    assert "12" in msg and "coordinated omission" in msg


def test_non_200s_are_surfaced_with_the_total():
    [msg] = run_health(summary(passes=90, fails=10))
    assert "10" in msg and "100" in msg


def test_percentiles_are_flagged_as_survivors_only():
    """The point of surfacing failures: the latency numbers exclude them."""
    [msg] = run_health(summary(passes=90, fails=10))
    assert "survivors" in msg


def test_both_conditions_report_independently():
    assert len(run_health(summary(dropped=5, passes=90, fails=10))) == 2


def test_missing_sections_do_not_raise():
    """A summary from a run that never started has neither section."""
    assert run_health({}) == []


def test_zero_token_run_does_not_crash_postprocessing():
    """A run where everything failed generates no tokens (rate 0). Duration and
    the request rate must be 0, not a ZeroDivisionError."""
    from metron.metrics import get_total_duration, calculate_common_requests

    summary = {
        "metrics": {"total_generated_tokens": {"rate": 0, "count": 0}},
        "root_group": {"checks": {"is status 200": {"passes": 0, "fails": 5}}},
    }
    assert get_total_duration(summary) == 0.0
    assert calculate_common_requests(summary) == 0.0


def test_content_free_successes_are_surfaced():
    [msg] = run_health(summary(empty=7, passes=20))
    assert "7" in msg and "no content" in msg


def test_estimated_token_counts_are_surfaced():
    [msg] = run_health(summary(estimated=20, passes=20))
    assert "20" in msg and "usage" in msg


def test_incomplete_streams_are_surfaced():
    [msg] = run_health(summary(incomplete=3, passes=20))
    assert "3" in msg and "truncated" in msg


def test_every_condition_reports_independently():
    s = summary(dropped=1, passes=1, fails=1, empty=1, estimated=1, incomplete=1)
    assert len(run_health(s)) == 5
