from __future__ import annotations

import json
import os
import re
import tempfile
from pathlib import Path
from typing import Any

import yaml

from cadmultiphysics.diagnostics import Diagnostic
from cadmultiphysics.errors import SchemaError

_BOOL = re.compile(r"^(?:true|True|TRUE|false|False|FALSE)$")


class _YamlLoader(yaml.SafeLoader):
    pass


_YamlLoader.yaml_implicit_resolvers = {
    key: list(value) for key, value in yaml.SafeLoader.yaml_implicit_resolvers.items()
}

for key, resolvers in list(_YamlLoader.yaml_implicit_resolvers.items()):
    _YamlLoader.yaml_implicit_resolvers[key] = [
        resolver for resolver in resolvers if resolver[0] != "tag:yaml.org,2002:bool"
    ]

for key in "tTfF":
    _YamlLoader.add_implicit_resolver("tag:yaml.org,2002:bool", _BOOL, [key])


def load_config(path: str | os.PathLike[str]) -> dict[str, Any]:
    source = Path(path)
    try:
        text = source.read_text(encoding="utf-8")
    except OSError as exc:
        raise SchemaError(
            [
                Diagnostic(
                    code="CONFIG_READ_FAILED",
                    message=f"Could not read '{source}'.",
                    path=(),
                    source="io",
                    backend_error=str(exc),
                )
            ]
        ) from exc
    try:
        if source.suffix.lower() == ".json":
            data = json.loads(text)
        else:
            data = yaml.load(text, Loader=_YamlLoader)
    except (json.JSONDecodeError, yaml.YAMLError) as exc:
        raise SchemaError(
            [
                Diagnostic(
                    code="CONFIG_PARSE_FAILED",
                    message=f"Could not parse '{source}'.",
                    path=(),
                    source="io",
                    backend_error=str(exc),
                )
            ]
        ) from exc
    if not isinstance(data, dict):
        raise SchemaError(
            [
                Diagnostic(
                    code="CONFIG_ROOT_INVALID",
                    message="Config root must be a mapping.",
                    path=(),
                    source="io",
                )
            ]
        )
    return data


def write_json(path: str | os.PathLike[str], payload: Any) -> None:
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp = tempfile.mkstemp(prefix=f".{target.name}.", suffix=".tmp", dir=target.parent)
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as handle:
            json.dump(payload, handle, indent=2, sort_keys=True)
            handle.write("\n")
        os.replace(tmp, target)
    except Exception:
        try:
            os.unlink(tmp)
        except OSError:
            pass
        raise
