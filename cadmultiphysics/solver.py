from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from pathlib import Path

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


def execute_solve(spec: ProblemSpec, plan: RunPlan, run_dir: Path) -> SolverResult:
    committed = initial_solution_state(spec, plan)
    trial = open_trial_state(committed, plan)
    diagnostic = Diagnostic(
        code="SOLVER_BACKEND_NOT_IMPLEMENTED",
        message=f"{spec.mode} solve execution requires the DOLFINx/UFL/PETSc backend.",
        path=("solver",),
        source="solver",
    )
    record = StepRecord(
        index=trial.trial_step or committed.committed_step + 1,
        status="failed",
        mode=spec.mode,
        solver=plan.solver,
        phases=("open", "predict", "discretize", "update", "build", "solve", "accept", "fail"),
        target_time=trial.time,
        time_unit=trial.time_unit,
        opened_state_hash=committed.state_hash,
        trial_state_hash=trial.state_hash,
        final_state_hash=committed.state_hash,
        acceptance=(
            AcceptanceCheck(
                name="petsc_converged",
                status="failed",
                diagnostic=diagnostic.code,
            ),
        ),
        diagnostics=(diagnostic,),
    )
    state_path = run_dir / "restarts" / "state_committed.json"
    step_path = run_dir / "logs" / f"step_{record.index:04d}.json"
    write_json(state_path, committed.model_dump(mode="json"))
    write_json(step_path, record.model_dump(mode="json"))
    return SolverResult(
        state=committed,
        steps=(record,),
        diagnostics=(diagnostic,),
        artifacts={"solution_state": state_path, f"step_{record.index:04d}": step_path},
        exit_code=2,
    )


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
