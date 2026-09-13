# Copyright (c) 2024 AXELTEC SOFTWARE LTD.
# SPDX-License-Identifier: Apache-2.0

from pathlib import Path
import uuid
import json


def create_artifact(base_dir: str = "artifacts") -> Path:
    """Create a fresh run directory under ``base_dir`` (relative to the cwd).

    Deliberately not relative to this file: an installed package would then
    write results into site-packages, where the user cannot find them and may
    not have permission to write.
    """
    path = Path(base_dir).expanduser().resolve() / str(uuid.uuid4())
    path.mkdir(parents=True, exist_ok=True)
    return path


def json_lines_to_common_json(filepath: Path) -> None:
    """
    By default, k6 stores traces in JSON-Lines format.
    This function converts JSON Lines file into common JSON
    by casting whole set of valid JSONs to list
    """
    common_json = []
    with open(filepath, "r") as json_file:
        for line in json_file.readlines():
            common_json.append(json.loads(line))

    with open(filepath, "w") as json_file:
        json.dump(common_json, json_file, indent=4)
