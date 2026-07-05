from __future__ import annotations

import argparse
import importlib.metadata
import json
import platform
import sys
from typing import Any

from cadmultiphysics import __version__
from cadmultiphysics.core import build_domain_ir, build_problem_spec, build_run_manifest, input_json_schema
from cadmultiphysics.diagnostics import Diagnostic, diagnostics_payload
from cadmultiphysics.errors import CadMPError
from cadmultiphysics.io import load_config, write_json
from cadmultiphysics.run import CommandResult, mesh, restart, solve


def main(argv: list[str] | None = None) -> int:
    parser = _parser()
    args = parser.parse_args(argv)
    try:
        return args.func(args)
    except CadMPError as exc:
        return _fail(exc.diagnostics, getattr(args, "json", False))


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="cadmp")
    subparsers = parser.add_subparsers(dest="command", required=True)
    validate = subparsers.add_parser("validate")
    validate.add_argument("problem")
    validate.add_argument("--json", action="store_true")
    validate.add_argument("--canonical", action="store_true")
    validate.add_argument("--manifest")
    validate.add_argument("--run-dir", default="run")
    validate.set_defaults(func=_validate)
    schema = subparsers.add_parser("schema")
    schema.add_argument("--format", choices=("json",), default="json")
    schema.set_defaults(func=_schema)
    inspect = subparsers.add_parser("inspect")
    inspect.add_argument("report")
    inspect.add_argument("--json", action="store_true")
    inspect.set_defaults(func=_inspect)
    version = subparsers.add_parser("version")
    version.add_argument("--full", action="store_true")
    version.set_defaults(func=_version)
    for name in ("mesh", "solve"):
        command = subparsers.add_parser(name)
        command.add_argument("problem")
        command.add_argument("--out", required=True)
        command.add_argument("--overwrite", action="store_true")
        command.add_argument("--json", action="store_true")
        command.set_defaults(func=_mesh if name == "mesh" else _solve)
    restart = subparsers.add_parser("restart")
    restart.add_argument("restart")
    restart.add_argument("--out", required=True)
    restart.add_argument("--overwrite", action="store_true")
    restart.add_argument("--json", action="store_true")
    restart.set_defaults(func=_restart)
    return parser


def _validate(args: argparse.Namespace) -> int:
    spec = build_problem_spec(load_config(args.problem))
    manifest = build_run_manifest(spec, args.run_dir)
    if args.manifest:
        write_json(args.manifest, manifest.model_dump(mode="json"))
    payload: dict[str, Any] = {
        "status": "ok",
        "name": spec.name,
        "mode": spec.mode,
        "content_hash": spec.content_hash,
        "domain": build_domain_ir(spec).model_dump(mode="json"),
    }
    if args.canonical:
        payload["spec"] = spec.model_dump(mode="json")
    if args.manifest:
        payload["manifest"] = str(args.manifest)
    if args.json:
        _print_json(payload)
    else:
        print(f"valid {spec.name} {spec.mode} {spec.content_hash}")
        if args.manifest:
            print(f"manifest {args.manifest}")
    return 0


def _schema(args: argparse.Namespace) -> int:
    _print_json(input_json_schema())
    return 0


def _inspect(args: argparse.Namespace) -> int:
    with open(args.report, "r", encoding="utf-8") as handle:
        payload = json.load(handle)
    if args.json:
        _print_json(payload)
        return 0
    status = payload.get("status", "<unknown>")
    command = payload.get("command", "<unknown>")
    name = payload.get("name", "<unknown>")
    domain = payload.get("domain") or {}
    mode = payload.get("mode") or domain.get("mode", "<unknown>")
    content_hash = payload.get("content_hash") or domain.get("content_hash", "<unknown>")
    artifacts = payload.get("artifacts") or {}
    diagnostics = payload.get("diagnostics") or []
    accepted_steps = payload.get("accepted_steps", 0)
    failed_steps = payload.get("failed_steps", 0)
    manifest = payload.get("manifest")
    print(f"status {status}")
    print(f"command {command}")
    print(f"name {name}")
    print(f"mode {mode}")
    print(f"content_hash {content_hash}")
    print(f"steps accepted={accepted_steps} failed={failed_steps}")
    print(f"artifacts {len(artifacts)}")
    if manifest:
        print(f"manifest {manifest}")
    print(f"diagnostics {len(diagnostics)}")
    for diagnostic in diagnostics[:5]:
        path = ".".join(str(part) for part in diagnostic.get("path", ())) if diagnostic.get("path") else "<root>"
        print(f"diagnostic {diagnostic.get('code', '<unknown>')} {path}")
    return 0


def _mesh(args: argparse.Namespace) -> int:
    return _emit_command_result(mesh(args.problem, args.out, args.overwrite), args)


def _solve(args: argparse.Namespace) -> int:
    return _emit_command_result(solve(args.problem, args.out, args.overwrite), args)


def _restart(args: argparse.Namespace) -> int:
    return _emit_command_result(restart(args.restart, args.out, args.overwrite), args)


def _version(args: argparse.Namespace) -> int:
    if not args.full:
        print(__version__)
        return 0
    _print_json(
        {
            "cadmultiphysics": __version__,
            "python": platform.python_version(),
            "packages": {
                name: _package_version(name)
                for name in (
                    "pydantic",
                    "pint",
                    "PyYAML",
                    "gmsh",
                    "fenics-dolfinx",
                    "fenics-basix",
                    "fenics-ffcx",
                    "fenics-ufl",
                    "mpi4py",
                    "mpich",
                    "petsc",
                    "petsc4py",
                    "adios2",
                )
            },
        }
    )
    return 0


def _emit_command_result(result: CommandResult, args: argparse.Namespace) -> int:
    payload = result.report.model_dump(mode="json")
    if args.json:
        _print_json(payload)
        return result.exit_code
    print(f"report {args.out}/report.json")
    if result.report.diagnostics:
        _fail(list(result.report.diagnostics), False, result.exit_code)
    return result.exit_code


def _fail(diagnostics: list[Diagnostic], json_mode: bool, exit_code: int = 1) -> int:
    if json_mode:
        _print_json(diagnostics_payload("error", diagnostics))
    else:
        for diagnostic in diagnostics:
            path = ".".join(str(part) for part in diagnostic.path) if diagnostic.path else "<root>"
            print(f"{diagnostic.severity} {diagnostic.code} {path}: {diagnostic.message}", file=sys.stderr)
            if diagnostic.hint:
                print(f"hint: {diagnostic.hint}", file=sys.stderr)
            if diagnostic.backend_error:
                print(f"backend: {diagnostic.backend_error}", file=sys.stderr)
    return exit_code


def _print_json(payload: Any) -> None:
    json.dump(payload, sys.stdout, indent=2, sort_keys=True)
    sys.stdout.write("\n")


def _package_version(name: str) -> str | None:
    try:
        return importlib.metadata.version(name)
    except importlib.metadata.PackageNotFoundError:
        return None
