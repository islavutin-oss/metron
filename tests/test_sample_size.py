# Copyright (c) 2024 AXELTEC SOFTWARE LTD.
# SPDX-License-Identifier: Apache-2.0
"""A percentile is only as good as the number of samples behind it.

metron decides which percentiles it will print from the request count, and
prints an em dash for the ones the run did not earn. That count came from
`http_reqs` — which handleSummary deletes, along with every other `http*`
metric, and it also deleted `total_requests_sent` outright. So the count was
always absent, n was always 0, and *every* p90 and p99 printed as an em dash no
matter how long the run was. The honesty feature reported nothing rather than
too much, which is the safe direction to fail and still wrong.

`total_requests_sent` now survives into the summary, and is read from there.
"""

import json
import pathlib

from metron.__main__ import supported_percentiles

JS = (pathlib.Path(__file__).resolve().parents[1] / "metron" / "metron.js").read_text()


def test_the_counter_survives_handle_summary():
    """It must be restored after the http* sweep, not before it."""
    body = JS[JS.index("export function handleSummary") :]
    assert (
        "delete data.metrics['total_requests_sent']" not in body
    ), "the request counter is deleted again"
    restore = body.index("data.metrics['total_requests_sent'] = requestsSent")
    sweep = body.index("key.startsWith('http')")
    assert (
        restore > sweep
    ), "the counter must be put back after the http* sweep, or it is removed again"


def test_it_is_captured_before_the_deletions():
    body = JS[JS.index("export function handleSummary") :]
    capture = body.index("const requestsSent")
    first_delete = body.index("delete data.metrics")
    assert capture < first_delete, "capture the counter before deleting anything"


def test_percentiles_need_their_samples():
    assert supported_percentiles(100)["p(90)"] is True
    assert supported_percentiles(99)["p(90)"] is False
    assert supported_percentiles(1000)["p(99)"] is True
    assert supported_percentiles(999)["p(99)"] is False


def test_no_samples_supports_nothing():
    """The old behaviour, now only reached when the count is genuinely absent."""
    ok = supported_percentiles(0)
    assert not any(ok.values())


def test_a_summary_carrying_the_counter_reads_its_n():
    summary = {
        "metrics": {
            "total_requests_sent": {"count": 1069, "rate": 71.2},
            "ttft": {"p(50)": 26.8, "p(90)": 35.5},
        }
    }
    metrics = summary["metrics"]
    reqs = metrics.get("total_requests_sent") or metrics.get("http_reqs") or {}
    assert int(reqs.get("count", 0)) == 1069
    assert supported_percentiles(1069)["p(99)"] is True


def test_the_json_shape_is_what_k6_writes_for_a_counter():
    """A k6 Counter serialises as count+rate; a Trend has no count at all."""
    counter = json.loads('{"count": 728, "rate": 16.2}')
    assert "count" in counter
