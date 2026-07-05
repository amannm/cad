from __future__ import annotations

import hashlib
import importlib.metadata
import importlib.util
import json
import math
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from time import perf_counter
from typing import Any

from cadmultiphysics.diagnostics import Diagnostic
from cadmultiphysics.io import write_json
from cadmultiphysics.schema import AcceptanceCheck, ProblemSpec, RunPlan, SolutionState, StepRecord


@dataclass(frozen=True)
class SolverResult:
    state: SolutionState
    steps: tuple[StepRecord, ...]
    diagnostics: tuple[Diagnostic, ...]
    artifacts: dict[str, Path]
    exit_code: int


@dataclass(frozen=True)
class BackendRequirement:
    import_name: str
    package_name: str
    role: str


class StepController:
    def __init__(self, spec: ProblemSpec, plan: RunPlan, run_dir: Path) -> None:
        self.spec = spec
        self.plan = plan
        self.run_dir = run_dir
        self.artifacts: dict[str, Path] = {}

    def run(self) -> SolverResult:
        started = perf_counter()
        self._ensure_dirs()
        committed = initial_solution_state(self.spec, self.plan)
        self._write_artifact("solution_state", self.run_dir / "restarts" / "state_committed.json", committed.model_dump(mode="json"))
        dependency_report = backend_dependency_report(self.spec, self.plan)
        self._write_artifact("backend_dependencies", self.run_dir / "logs" / "backend_dependencies.json", dependency_report)
        self._write_artifact("field_output_plan", self.run_dir / "fields" / "output_plan.json", field_output_plan(self.spec, self.plan))
        self._write_artifact("restart_plan", self.run_dir / "restarts" / "restart_plan.json", restart_plan(self.spec, self.plan))
        steps: list[StepRecord] = []
        diagnostics: list[Diagnostic] = []
        missing = tuple(item for item in dependency_report["dependencies"] if not item["available"])
        while committed.committed_step < self.plan.steps:
            trial = open_trial_state(committed, self.plan)
            self._write_artifact(f"trial_state_{trial.trial_step:04d}", self.run_dir / "logs" / f"state_trial_{trial.trial_step:04d}.json", trial.model_dump(mode="json"))
            if missing:
                record = self._missing_backend_step(committed, trial, missing)
            else:
                record = self._unimplemented_backend_step(committed, trial)
            steps.append(record)
            diagnostics.extend(record.diagnostics)
            self._write_artifact(f"step_{record.index:04d}", self.run_dir / "logs" / f"step_{record.index:04d}.json", record.model_dump(mode="json"))
            if record.status == "failed":
                break
            committed = commit_trial_state(trial)
            self._write_artifact("solution_state", self.run_dir / "restarts" / "state_committed.json", committed.model_dump(mode="json"))
        trace = {
            "status": "failed" if diagnostics else "ok",
            "steps_planned": self.plan.steps,
            "steps_recorded": len(steps),
            "accepted_steps": sum(1 for step in steps if step.status == "accepted"),
            "failed_steps": sum(1 for step in steps if step.status == "failed"),
            "final_state_hash": committed.state_hash,
            "elapsed_seconds": perf_counter() - started,
        }
        self._write_artifact("solver_trace", self.run_dir / "logs" / "solver_trace.json", trace)
        return SolverResult(
            state=committed,
            steps=tuple(steps),
            diagnostics=tuple(diagnostics),
            artifacts=dict(self.artifacts),
            exit_code=1 if diagnostics else 0,
        )

    def _missing_backend_step(self, committed: SolutionState, trial: SolutionState, missing: tuple[dict[str, Any], ...]) -> StepRecord:
        names = tuple(item["import_name"] for item in missing)
        diagnostic = Diagnostic(
            code="SOLVER_BACKEND_DEPENDENCY_MISSING",
            message=f"Solve execution requires missing backend packages: {', '.join(names)}.",
            path=("solver", "backend"),
            step_index=trial.trial_step,
            source="solver",
            payload={"missing": missing, "driver": self.plan.solver, "required_interface": self.plan.contract.physics_interface},
        )
        return self._failed_step(
            committed,
            trial,
            ("open", "predict", "discretize", "update", "build", "fail"),
            (
                AcceptanceCheck(
                    name="backend_dependencies",
                    status="failed",
                    diagnostic=diagnostic.code,
                    payload={"missing": names},
                ),
                AcceptanceCheck(
                    name="committed_state_unchanged",
                    status="passed",
                    payload={"state_hash": committed.state_hash},
                ),
            ),
            (diagnostic,),
        )

    def _unimplemented_backend_step(self, committed: SolutionState, trial: SolutionState) -> StepRecord:
        finite = finite_state_diagnostics(trial, trial.trial_step or committed.committed_step + 1)
        if finite:
            return self._failed_step(
                committed,
                trial,
                ("open", "predict", "discretize", "update", "build", "solve", "accept", "fail"),
                (
                    AcceptanceCheck(name="backend_dependencies", status="passed"),
                    AcceptanceCheck(name="finite_state", status="failed", diagnostic=finite[0].code),
                    AcceptanceCheck(name="committed_state_unchanged", status="passed", payload={"state_hash": committed.state_hash}),
                ),
                finite,
            )
        diagnostic = Diagnostic(
            code="SOLVER_BACKEND_NOT_IMPLEMENTED",
            message=f"{self.spec.mode} solve execution requires the DOLFINx/UFL/PETSc backend driver.",
            path=("solver",),
            step_index=trial.trial_step,
            source="solver",
            payload={"required_interface": self.plan.contract.physics_interface, "driver": self.plan.solver},
        )
        return self._failed_step(
            committed,
            trial,
            ("open", "predict", "discretize", "update", "build", "solve", "accept", "fail"),
            (
                AcceptanceCheck(name="backend_dependencies", status="passed"),
                AcceptanceCheck(name="finite_state", status="passed"),
                AcceptanceCheck(
                    name="petsc_converged",
                    status="failed",
                    diagnostic=diagnostic.code,
                    payload={"driver": self.plan.solver},
                ),
                AcceptanceCheck(
                    name="committed_state_unchanged",
                    status="passed",
                    payload={"state_hash": committed.state_hash},
                ),
            ),
            (diagnostic,),
        )

    def _failed_step(
        self,
        committed: SolutionState,
        trial: SolutionState,
        phases: tuple[str, ...],
        acceptance: tuple[AcceptanceCheck, ...],
        diagnostics: tuple[Diagnostic, ...],
    ) -> StepRecord:
        return StepRecord(
            index=trial.trial_step or committed.committed_step + 1,
            status="failed",
            mode=self.spec.mode,
            solver=self.plan.solver,
            phases=phases,
            target_time=trial.time,
            time_unit=trial.time_unit,
            opened_state_hash=committed.state_hash,
            trial_state_hash=trial.state_hash,
            final_state_hash=committed.state_hash,
            acceptance=acceptance,
            diagnostics=diagnostics,
        )

    def _write_artifact(self, name: str, path: Path, payload: Any) -> None:
        write_json(path, payload)
        self.artifacts[name] = path

    def _ensure_dirs(self) -> None:
        for path in (self.run_dir, self.run_dir / "fields", self.run_dir / "restarts", self.run_dir / "logs"):
            path.mkdir(parents=True, exist_ok=True)


def execute_solve(spec: ProblemSpec, plan: RunPlan, run_dir: Path) -> SolverResult:
    return StepController(spec, plan, run_dir).run()


def initial_solution_state(spec: ProblemSpec, plan: RunPlan) -> SolutionState:
    time = plan.time.start if plan.time else None
    unit = plan.time.unit if plan.time else None
    return _hashed_state(
        SolutionState(
            schema_version=spec.version,
            content_hash=spec.content_hash,
            mode=spec.mode,
            fields=tuple(field.name for field in spec.fields),
            committed_step=0,
            time=time,
            time_unit=unit,
        )
    )


def open_trial_state(committed: SolutionState, plan: RunPlan) -> SolutionState:
    trial_step = committed.committed_step + 1
    target_time = _target_time(plan, trial_step)
    return _hashed_state(
        committed.model_copy(
            update={
                "trial_step": trial_step,
                "time": target_time,
                "time_unit": plan.time.unit if plan.time else None,
                "state_hash": "",
            }
        )
    )


def commit_trial_state(trial: SolutionState) -> SolutionState:
    step = trial.trial_step or trial.committed_step
    return _hashed_state(
        trial.model_copy(
            update={
                "committed_step": step,
                "trial_step": None,
                "restart_markers": dict(sorted({**trial.restart_markers, f"step_{step:04d}": trial.state_hash}.items())),
                "state_hash": "",
            }
        )
    )


def backend_dependency_report(spec: ProblemSpec, plan: RunPlan) -> dict[str, Any]:
    dependencies = tuple(_dependency_payload(requirement) for requirement in backend_requirements(spec, plan))
    return {
        "driver": plan.solver,
        "mode": spec.mode,
        "problem_kind": plan.problem_kind,
        "required_interface": plan.contract.physics_interface,
        "dependencies": dependencies,
        "ready": all(item["available"] for item in dependencies),
    }


def backend_requirements(spec: ProblemSpec, plan: RunPlan) -> tuple[BackendRequirement, ...]:
    requirements = [
        BackendRequirement("dolfinx", "fenics-dolfinx", "mesh conversion, function spaces, assembly, VTX writer"),
        BackendRequirement("ufl", "fenics-ufl", "weak forms and symbolic residuals"),
        BackendRequirement("petsc4py", "petsc4py", f"{plan.solver.upper()} solver driver and convergence reasons"),
    ]
    if spec.output.format == "vtx":
        requirements.append(BackendRequirement("adios2", "adios2", "ADIOS2/BP field output"))
    return tuple(requirements)


def field_output_plan(spec: ProblemSpec, plan: RunPlan) -> dict[str, Any]:
    return {
        "format": spec.output.format,
        "fields": spec.output.fields,
        "derived_fields": spec.output.derived_fields,
        "cadence": spec.output.cadence,
        "steps": plan.output_steps,
        "writer_options": spec.output.writer_options,
    }


def restart_plan(spec: ProblemSpec, plan: RunPlan) -> dict[str, Any]:
    return {
        "schema_version": spec.version,
        "content_hash": spec.content_hash,
        "cadence": plan.restart_cadence,
        "steps": plan.restart_steps,
        "checkpoint_policy": plan.checkpoint_policy,
        "field_layout": tuple(field.name for field in spec.fields),
    }


def finite_state_diagnostics(state: SolutionState, step_index: int | None = None) -> tuple[Diagnostic, ...]:
    diagnostics: list[Diagnostic] = []
    for root in ("field_state", "history_state", "material_state"):
        diagnostics.extend(_finite_diagnostics(getattr(state, root), (root,), step_index))
    return tuple(diagnostics)


def _finite_diagnostics(value: Any, path: tuple[str | int, ...], step_index: int | None) -> list[Diagnostic]:
    diagnostics: list[Diagnostic] = []
    if isinstance(value, Mapping):
        for key, item in sorted(value.items()):
            diagnostics.extend(_finite_diagnostics(item, (*path, str(key)), step_index))
    elif isinstance(value, Sequence) and not isinstance(value, str | bytes):
        for index, item in enumerate(value):
            diagnostics.extend(_finite_diagnostics(item, (*path, index), step_index))
    elif isinstance(value, int | float) and not isinstance(value, bool) and not math.isfinite(float(value)):
        diagnostics.append(
            Diagnostic(
                code="SOLVER_STATE_NONFINITE",
                message="Solution state contains a non-finite numeric value.",
                path=path,
                step_index=step_index,
                source="solver",
            )
        )
    return diagnostics


def _dependency_payload(requirement: BackendRequirement) -> dict[str, Any]:
    return {
        "import_name": requirement.import_name,
        "package_name": requirement.package_name,
        "role": requirement.role,
        "available": _module_available(requirement.import_name),
        "version": _version(requirement.package_name),
    }


def _module_available(name: str) -> bool:
    try:
        return importlib.util.find_spec(name) is not None
    except (ImportError, ValueError):
        return False


def _version(package: str) -> str | None:
    try:
        return importlib.metadata.version(package)
    except importlib.metadata.PackageNotFoundError:
        return None


def _target_time(plan: RunPlan, step: int) -> float | None:
    if plan.time is None:
        return None
    return min(plan.time.stop, plan.time.start + step * plan.time.step)


def _hashed_state(state: SolutionState) -> SolutionState:
    return state.model_copy(update={"state_hash": _state_hash(state)})


def _state_hash(state: SolutionState) -> str:
    payload = state.model_dump(mode="json", exclude={"state_hash"})
    encoded = json.dumps(payload, sort_keys=True, separators=(",", ":"), ensure_ascii=True).encode()
    return hashlib.sha256(encoded).hexdigest()
