from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Literal

from pydantic import ValidationError

from cadmultiphysics.core import build_domain_ir, build_problem_spec, build_run_manifest, build_run_plan
from cadmultiphysics.diagnostics import Diagnostic, schema_diagnostics
from cadmultiphysics.errors import MeshError
from cadmultiphysics.io import load_config, write_diagnostics_csv, write_json
from cadmultiphysics.mesh import generate_mesh
from cadmultiphysics.schema import DomainIR, MeshMetadata, ProblemSpec, RestartState, RunArtifact, RunManifest, RunPlan, RunReport, SolutionState, StepRecord
from cadmultiphysics.solver import execute_solve
from cadmultiphysics.units import QuantitySpec


@dataclass(frozen=True)
class CommandResult:
    report: RunReport
    exit_code: int


def mesh(problem: str, out: str) -> CommandResult:
    return mesh_spec(build_problem_spec(load_config(problem)), out)


def mesh_spec(spec: ProblemSpec, out: str) -> CommandResult:
    run_dir = Path(out).resolve()
    domain, plan, manifest, artifacts = _problem_artifacts(spec, run_dir, "mesh")
    diagnostics: tuple[Diagnostic, ...] = ()
    try:
        build = generate_mesh(spec, run_dir / "mesh")
        _write_mesh_artifacts(build.metadata, build.artifacts)
        artifacts.update({name: _artifact(path) for name, path in build.artifacts.items()})
        manifest = _write_manifest(spec, run_dir, artifacts)
        artifacts["manifest"] = _artifact(run_dir / "manifest.json")
        restart_path = run_dir / "restarts" / "restart_0000.json"
        restart_state = _mesh_restart_state(spec, manifest, artifacts["manifest"].sha256)
        write_json(restart_path, restart_state.model_dump(mode="json"))
        artifacts["restart_0000"] = _artifact(restart_path)
    except MeshError as exc:
        diagnostics = tuple(exc.diagnostics)
    return _finish_problem_command("mesh", spec, domain, plan, manifest, artifacts, diagnostics, run_dir, 1 if diagnostics else 0)


def solve(problem: str, out: str) -> CommandResult:
    return solve_spec(build_problem_spec(load_config(problem)), out)


def solve_spec(spec: ProblemSpec, out: str) -> CommandResult:
    run_dir = Path(out).resolve()
    domain, plan, manifest, artifacts = _problem_artifacts(spec, run_dir, "solve")
    result = execute_solve(spec, plan, run_dir)
    artifacts.update({name: _artifact(path) for name, path in result.artifacts.items()})
    manifest = _write_manifest(spec, run_dir, artifacts)
    artifacts["manifest"] = _artifact(run_dir / "manifest.json")
    restart_path = run_dir / "restarts" / "restart_0000.json"
    restart_state = _solve_restart_state(spec, manifest, artifacts["manifest"].sha256, result.state)
    write_json(restart_path, restart_state.model_dump(mode="json"))
    artifacts["restart_0000"] = _artifact(restart_path)
    return _finish_problem_command("solve", spec, domain, plan, manifest, artifacts, result.diagnostics, run_dir, result.exit_code, result.state, result.steps)


def restart(restart_path: str, out: str) -> CommandResult:
    run_dir = Path(out).resolve()
    _ensure_run_dirs(run_dir)
    source = Path(restart_path)
    artifacts: dict[str, RunArtifact] = {}
    state = _load_restart_state(source)
    if isinstance(state, tuple):
        _write_diagnostics_artifact(run_dir, artifacts, state)
        report = _restart_report(None, artifacts, state)
        write_json(run_dir / "report.json", report.model_dump(mode="json"))
        return CommandResult(report, 1)
    state_path = run_dir / "restart_state.json"
    write_json(state_path, state.model_dump(mode="json"))
    artifacts["restart_state"] = _artifact(state_path)
    diagnostics = _restart_diagnostics(state, source)
    if not diagnostics:
        diagnostics = (
            Diagnostic(
                code="RESTART_BACKEND_NOT_IMPLEMENTED",
                message="Restart solve execution is not implemented.",
                path=("restart",),
                source="restart",
            ),
        )
    _write_diagnostics_artifact(run_dir, artifacts, diagnostics)
    report = _restart_report(state, artifacts, diagnostics)
    write_json(run_dir / "report.json", report.model_dump(mode="json"))
    return CommandResult(report, 2)


def _problem_artifacts(
    spec: ProblemSpec,
    run_dir: Path,
    command: Literal["mesh", "solve"],
) -> tuple[DomainIR, RunPlan, RunManifest, dict[str, RunArtifact]]:
    _ensure_run_dirs(run_dir)
    domain = build_domain_ir(spec)
    plan = build_run_plan(spec)
    files = {
        "spec": run_dir / "spec.json",
        "domain": run_dir / "domain.json",
        "run_plan": run_dir / "run_plan.json",
        "mesh_plan": run_dir / "mesh" / "mesh_plan.json",
    }
    write_json(files["spec"], spec.model_dump(mode="json"))
    write_json(files["domain"], domain.model_dump(mode="json"))
    write_json(files["run_plan"], plan.model_dump(mode="json"))
    write_json(files["mesh_plan"], spec.mesh.model_dump(mode="json"))
    if command == "solve":
        files["solver_profile"] = run_dir / "logs" / "solver_profile.json"
        write_json(files["solver_profile"], spec.solver.model_dump(mode="json"))
    artifacts = {name: _artifact(path) for name, path in files.items()}
    manifest = _write_manifest(spec, run_dir, artifacts)
    artifacts["manifest"] = _artifact(run_dir / "manifest.json")
    return domain, plan, manifest, artifacts


def _write_manifest(spec: ProblemSpec, run_dir: Path, artifacts: dict[str, RunArtifact]) -> RunManifest:
    manifest = build_run_manifest(spec, str(run_dir)).model_copy(
        update={"artifact_hashes": {name: artifact.sha256 for name, artifact in artifacts.items() if name != "manifest" and not name.startswith("restart_")}}
    )
    write_json(run_dir / "manifest.json", manifest.model_dump(mode="json"))
    return manifest


def _write_mesh_artifacts(metadata: MeshMetadata, artifacts: dict[str, Path]) -> None:
    write_json(artifacts["mesh_metadata"], metadata.model_dump(mode="json"))
    write_json(artifacts["tag_map"], metadata.tags.model_dump(mode="json"))
    write_json(artifacts["geometry_ir"], [entity.model_dump(mode="json") for entity in metadata.entities])


def _mesh_restart_state(spec: ProblemSpec, manifest: RunManifest, manifest_hash: str) -> RestartState:
    return RestartState(
        schema_version=spec.version,
        content_hash=spec.content_hash,
        manifest_path=manifest.output_paths["manifest"],
        manifest_hash=manifest_hash,
        mode=spec.mode,
        fields=tuple(field.name for field in spec.fields),
        step_index=0,
        time=spec.time.start if spec.time else None,
        artifacts={"mesh": manifest.output_paths["mesh"]},
    )


def _solve_restart_state(spec: ProblemSpec, manifest: RunManifest, manifest_hash: str, state: SolutionState) -> RestartState:
    return RestartState(
        schema_version=spec.version,
        content_hash=spec.content_hash,
        manifest_path=manifest.output_paths["manifest"],
        manifest_hash=manifest_hash,
        mode=spec.mode,
        fields=state.fields,
        step_index=state.committed_step,
        time=_state_time(state),
        state_hash=state.state_hash,
        artifacts={"state": str(Path(manifest.output_paths["restarts"]) / "state_committed.json")},
    )


def _finish_problem_command(
    command: Literal["mesh", "solve"],
    spec: ProblemSpec,
    domain: DomainIR,
    plan: RunPlan,
    manifest: RunManifest,
    artifacts: dict[str, RunArtifact],
    diagnostics: tuple[Diagnostic, ...],
    run_dir: Path,
    exit_code: int,
    state: SolutionState | None = None,
    steps: tuple[StepRecord, ...] = (),
) -> CommandResult:
    _write_diagnostics_artifact(run_dir, artifacts, diagnostics)
    report = RunReport(
        command=command,
        status="failed" if diagnostics else "ok",
        name=spec.name,
        mode=spec.mode,
        content_hash=spec.content_hash,
        accepted_steps=sum(1 for step in steps if step.status == "accepted"),
        failed_steps=sum(1 for step in steps if step.status == "failed"),
        artifact_count=len(artifacts),
        domain=domain,
        run_plan=plan,
        manifest=manifest.output_paths["manifest"],
        manifest_hash=artifacts["manifest"].sha256,
        backend_versions=manifest.backend_versions,
        mpi_size=manifest.mpi_size,
        artifacts=artifacts,
        state=state,
        steps=steps,
        diagnostics=diagnostics,
    )
    write_json(run_dir / "report.json", report.model_dump(mode="json"))
    return CommandResult(report, exit_code)


def _write_diagnostics_artifact(run_dir: Path, artifacts: dict[str, RunArtifact], diagnostics: tuple[Diagnostic, ...]) -> None:
    path = run_dir / "diagnostics.csv"
    write_diagnostics_csv(path, diagnostics)
    artifacts["diagnostics"] = _artifact(path)


def _state_time(state: SolutionState) -> QuantitySpec | None:
    if state.time is None:
        return None
    return QuantitySpec(magnitude=state.time, unit=state.time_unit or "second", dimension="[time]")


def _load_restart_state(source: Path) -> RestartState | tuple[Diagnostic, ...]:
    try:
        payload = json.loads(source.read_text(encoding="utf-8"))
    except OSError as exc:
        return (
            Diagnostic(
                code="RESTART_READ_FAILED",
                message=f"Could not read '{source}'.",
                path=("restart",),
                source="restart",
                backend_error=str(exc),
            ),
        )
    except json.JSONDecodeError as exc:
        return (
            Diagnostic(
                code="RESTART_PARSE_FAILED",
                message=f"Could not parse '{source}'.",
                path=("restart",),
                source="restart",
                backend_error=str(exc),
            ),
        )
    try:
        return RestartState.model_validate(payload)
    except ValidationError as exc:
        return tuple(
            Diagnostic(
                code="RESTART_INVALID",
                message=diagnostic.message,
                path=diagnostic.path,
                source="restart",
                backend_error=diagnostic.backend_error,
            )
            for diagnostic in schema_diagnostics(exc)
        )


def _restart_diagnostics(state: RestartState, source: Path) -> tuple[Diagnostic, ...]:
    manifest_path = Path(state.manifest_path)
    if not manifest_path.is_absolute():
        manifest_path = source.parent / manifest_path
    if not manifest_path.exists():
        return (
            Diagnostic(
                code="RESTART_MANIFEST_MISSING",
                message=f"Restart manifest '{manifest_path}' does not exist.",
                path=("manifest_path",),
                source="restart",
            ),
        )
    observed = _sha256(manifest_path)
    if observed != state.manifest_hash:
        return (
            Diagnostic(
                code="RESTART_MANIFEST_HASH_MISMATCH",
                message="Restart manifest hash does not match restart metadata.",
                path=("manifest_hash",),
                source="restart",
                payload={"expected": state.manifest_hash, "observed": observed},
            ),
        )
    return ()


def _restart_report(
    state: RestartState | None,
    artifacts: dict[str, RunArtifact],
    diagnostics: tuple[Diagnostic, ...],
) -> RunReport:
    return RunReport(
        command="restart",
        status="failed" if diagnostics else "ok",
        name=None,
        mode=state.mode if state else None,
        content_hash=state.content_hash if state else None,
        artifact_count=len(artifacts),
        domain=None,
        run_plan=None,
        manifest=state.manifest_path if state else None,
        manifest_hash=state.manifest_hash if state else None,
        artifacts=artifacts,
        diagnostics=diagnostics,
    )


def _ensure_run_dirs(run_dir: Path) -> None:
    for path in (run_dir, run_dir / "mesh", run_dir / "fields", run_dir / "restarts", run_dir / "logs"):
        path.mkdir(parents=True, exist_ok=True)


def _artifact(path: Path) -> RunArtifact:
    return RunArtifact(path=str(path), sha256=_sha256(path))


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()
