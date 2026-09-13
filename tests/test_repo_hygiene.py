# Copyright (c) 2024 AXELTEC SOFTWARE LTD.
# SPDX-License-Identifier: Apache-2.0

"""This repository is public. Names, half-finished notes and anything
secret-shaped must not ride along in it."""

import re
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def _sources():
    for pat in ("*.py", "*.js", "*.md", "*.toml", "Dockerfile*"):
        for p in ROOT.rglob(pat):
            s = str(p)
            if "node_modules" in s or "/.venv/" in s or "/.git/" in s:
                continue
            if p.name == "test_repo_hygiene.py":
                continue  # this file names the very things it forbids
            yield p


def test_no_notes_to_self():
    hits = []
    for p in _sources():
        for n, line in enumerate(p.read_text(errors="ignore").splitlines(), 1):
            if re.search(r"\b(TODO|FIXME|HACK|XXX)\b", line):
                hits.append(f"{p.name}:{n}")
    assert not hits, f"unfinished-work markers in a public repo: {hits}"


# A public contact address is meant to be here; anything else is someone's
# personal mail arriving by accident.
PUBLIC_CONTACT = "info@axel-t.com"

PATTERNS = {
    "an email address": re.compile(
        r"\b[A-Za-z0-9._%%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}\b"
    ),
    "a home directory path": re.compile(r"(?:/home/|/Users/|/root/)[A-Za-z0-9._-]+"),
    "a private network address": re.compile(
        # All four octets required: a version like "10.8.2" in a README is not
        # an address, and flagging it teaches people to ignore this test.
        r"\b(?:10\.\d{1,3}|192\.168|172\.(?:1[6-9]|2\d|3[01])|100\.(?:6[4-9]|[7-9]\d|1[01]\d|12[0-7]))"
        r"\.\d{1,3}\.\d{1,3}\b"
    ),
    "a credential": re.compile(
        r"\b(?:sk-[A-Za-z0-9_-]{16,}|pk_[A-Za-z0-9]{16,}|gh[pousr]_[A-Za-z0-9]{20,}"
        r"|AKIA[0-9A-Z]{16}|-----BEGIN [A-Z ]*PRIVATE KEY-----)"
    ),
}


def test_nothing_identifying_rides_along():
    """Detect the *shape* of a leak, not a list of words.

    This used to be a denylist of names — which only ever catches what someone
    already thought of, goes stale the moment a new project exists, and puts
    the very roster it protects into a file anyone can read. These patterns
    catch a stranger's email, a path out of someone's home directory, an
    internal address and a credential, none of which anyone has to remember to
    add.
    """
    hits = []
    for p in _sources():
        text = p.read_text(errors="ignore")
        for label, rx in PATTERNS.items():
            for m in rx.finditer(text):
                found = m.group(0)
                if label == "an email address" and found.lower() == PUBLIC_CONTACT:
                    continue
                line = text[: m.start()].count("\n") + 1
                hits.append(f"{label}: {found!r} at {p.name}:{line}")
    assert not hits, "identifying details in a public repo:\n  " + "\n  ".join(hits)
