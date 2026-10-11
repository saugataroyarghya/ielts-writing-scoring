"""Verify saved notebook source bundles without executing notebook code."""

import ast
import base64
import hashlib
import io
import json
import zipfile
from pathlib import Path, PurePosixPath

ROOT = Path(__file__).resolve().parents[1]


def main():
    manifest = json.loads((ROOT / "docs/training_sources.json").read_text())
    for run in manifest["runs"]:
        notebook = json.loads((ROOT / run["notebook"]).read_text())
        bundles = []
        for cell in notebook["cells"]:
            if cell["cell_type"] != "code":
                continue
            tree = ast.parse("".join(cell["source"]))
            for node in ast.walk(tree):
                if (isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute)
                        and node.func.attr == "b64decode" and node.args):
                    bundles.append(base64.b64decode(ast.literal_eval(node.args[0]), validate=True))
        if len(bundles) != 1:
            raise ValueError(f"Expected one embedded source bundle: {run['notebook']}")
        payload = bundles[0]
        if hashlib.sha256(payload).hexdigest() != run["source_bundle_sha256"]:
            raise ValueError(f"Source bundle hash mismatch: {run['name']}")
        with zipfile.ZipFile(io.BytesIO(payload)) as archive:
            names = archive.namelist()
            if len(names) != len(set(names)):
                raise ValueError("Duplicate source bundle members")
            if set(names) != set(run["files_sha256"]):
                raise ValueError("Source bundle member list changed")
            for name in names:
                path = PurePosixPath(name)
                if path.is_absolute() or ".." in path.parts:
                    raise ValueError("Unsafe source bundle path")
                original = archive.read(name)
                if hashlib.sha256(original).hexdigest() != run["files_sha256"][name]:
                    raise ValueError(f"Source member hash mismatch: {name}")
                if (ROOT / name).read_bytes() != original:
                    raise ValueError(f"Current file differs from the recorded training source: {name}")
        print(f"{run['name']}: {len(names)} source files match the completed run")


if __name__ == "__main__":
    main()
