# Copyright (c) 2024 AXELTEC SOFTWARE LTD.
# SPDX-License-Identifier: Apache-2.0

import re
import shutil
import subprocess
import textwrap
from pathlib import Path

import pytest

SCRIPT = Path(__file__).resolve().parents[1] / "metron" / "metron.js"


def _make_rng_source():
    """Lift makeRng out of the k6 script and check it, not a copy of it."""
    text = SCRIPT.read_text()
    m = re.search(r"function makeRng\(a\) \{.*?\n\}", text, re.S)
    assert m, "makeRng not found in metron.js — did the seeding change shape?"
    return m.group(0)


requires_node = pytest.mark.skipif(
    shutil.which("node") is None, reason="node not installed"
)


def _run(js):
    out = subprocess.run(["node", "-e", js], capture_output=True, text=True)
    assert out.returncode == 0, out.stderr
    return out.stdout.strip()


@requires_node
def test_the_same_seed_gives_the_same_sequence():
    js = _make_rng_source() + textwrap.dedent("""
        const a = makeRng(42), b = makeRng(42);
        console.log([a(), a(), a()].join(',') === [b(), b(), b()].join(','));
    """)
    assert _run(js) == "true"


@requires_node
def test_different_seeds_diverge():
    js = _make_rng_source() + textwrap.dedent("""
        const a = makeRng(1), b = makeRng(2);
        console.log(a() !== b());
    """)
    assert _run(js) == "true"


@requires_node
def test_output_is_in_the_unit_interval():
    js = _make_rng_source() + textwrap.dedent("""
        const r = makeRng(7);
        let ok = true;
        for (let i = 0; i < 1000; i++) { const x = r(); if (x < 0 || x >= 1) ok = false; }
        console.log(ok);
    """)
    assert _run(js) == "true"


def test_the_script_no_longer_calls_unseeded_random():
    """Math.random() anywhere in the request path reintroduces the bug the
    seed was added to fix."""
    body = SCRIPT.read_text()
    body = body[body.index("export default function") :]
    assert "Math.random()" not in body


def test_seed_is_a_documented_cli_flag():
    """The seed is user-controllable, not hardcoded: it must show in --help."""
    out = subprocess.run(
        ["python3", "-m", "metron", "--help"], capture_output=True, text=True
    )
    assert "--seed" in out.stdout
