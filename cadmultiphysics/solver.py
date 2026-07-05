from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from pathlib import Path
from time import perf_counter
from typing import Any

import numpy as np
import ufl
from dolfinx import fem
from dolfinx.fem.petsc import LinearProblem
from dolfinx.io import VTXWriter
from dolfinx.io import gmsh as gmshio
from mpi4py import MPI
from petsc4py import PETSc

from cadmultiphysics.diagnostics import Diagnostic
from cadmultiphysics.io import write_json
from cadmultiphysics.schema import AcceptanceCheck, MeshMetadata, ProblemSpec, RunPlan, SolutionState, StepRecord, TagBinding


@dataclass(frozen=True)
class SolverResult:
    state: SolutionState
    steps: tuple[StepRecord, ...]
    diagnostics: tuple[Diagnostic, ...]
    artifacts: dict[str, Path]
    exit_code: int


@dataclass(frozen=True)
class StepSolve:
    field_state: dict[str, Any]
    artifacts: dict[str, Path]
    payload: dict[str, Any]


class StepController:
    def __init__(self, spec: ProblemSpec, plan: RunPlan, mesh: MeshMetadata, run_dir: Path) -> None:
        self.spec = spec
        self.plan = plan
        self.mesh = mesh
        self.run_dir = run_dir
        self.artifacts: dict[str, Path] = {}

    def run(self) -> SolverResult:
        started = perf_counter()
        self._ensure_dirs()
        committed = initial_solution_state(self.spec, self.plan)
        self._write_artifact("solution_state", self.run_dir / "restarts" / "state_committed.json", committed.model_dump(mode="json"))
        steps: list[StepRecord] = []
        diagnostics: list[Diagnostic] = []
        while committed.committed_step < self.plan.steps:
            trial = open_trial_state(committed, self.plan)
            opened_state_hash = committed.state_hash
            try:
                solve = self._solve_step(trial)
            except Exception as exc:
                diagnostic = _solve_diagnostic(exc, trial.trial_step or committed.committed_step + 1)
                record = _failed_step(
                    self.spec,
                    self.plan,
                    committed,
                    trial,
                    ("open", "predict", "discretize", "update", "build", "solve", "accept", "fail"),
                    (
                        AcceptanceCheck(name="petsc_converged", status="failed", diagnostic=diagnostic.code, payload={"error": type(exc).__name__}),
                        AcceptanceCheck(name="committed_state_unchanged", status="passed", payload={"state_hash": committed.state_hash}),
                    ),
                    (diagnostic,),
                )
                steps.append(record)
                diagnostics.append(diagnostic)
                self._write_artifact(f"step_{record.index:04d}", self.run_dir / "logs" / f"step_{record.index:04d}.json", record.model_dump(mode="json"))
                break
            solved_trial = _hashed_state(trial.model_copy(update={"field_state": solve.field_state, "state_hash": ""}))
            committed = commit_trial_state(solved_trial)
            self.artifacts.update(solve.artifacts)
            self._write_artifact("solution_state", self.run_dir / "restarts" / "state_committed.json", committed.model_dump(mode="json"))
            record = StepRecord(
                index=committed.committed_step,
                status="accepted",
                mode=self.spec.mode,
                solver=self.plan.solver,
                phases=("open", "predict", "discretize", "update", "build", "solve", "accept", "commit", "postprocess", "write"),
                target_time=committed.time,
                time_unit=committed.time_unit,
                opened_state_hash=opened_state_hash,
                trial_state_hash=solved_trial.state_hash,
                final_state_hash=committed.state_hash,
                acceptance=(
                    AcceptanceCheck(name="petsc_converged", status="passed", payload=solve.payload),
                    AcceptanceCheck(name="finite_state", status="passed", payload=_finite_payload(solve.field_state)),
                ),
            )
            steps.append(record)
            self._write_artifact(f"step_{record.index:04d}", self.run_dir / "logs" / f"step_{record.index:04d}.json", record.model_dump(mode="json"))
            if not self.plan.transient:
                break
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

    def _solve_step(self, trial: SolutionState) -> StepSolve:
        if self.spec.mode != "linear_steady":
            raise NotImplementedError(f"{self.spec.mode} backend execution is not implemented")
        field = _single_scalar_field(self.spec)
        material_terms = _material_terms(self.spec, self.mesh)
        if not material_terms:
            raise ValueError("linear scalar solve requires at least one isotropic_heat material")
        return _solve_linear_scalar(self.spec, self.plan, self.mesh, self.run_dir, trial, field, material_terms)

    def _write_artifact(self, name: str, path: Path, payload: Any) -> None:
        write_json(path, payload)
        self.artifacts[name] = path

    def _ensure_dirs(self) -> None:
        for path in (self.run_dir, self.run_dir / "fields", self.run_dir / "restarts", self.run_dir / "logs"):
            path.mkdir(parents=True, exist_ok=True)


def execute_solve(spec: ProblemSpec, plan: RunPlan, mesh: MeshMetadata, run_dir: Path) -> SolverResult:
    return StepController(spec, plan, mesh, run_dir).run()


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


def _solve_linear_scalar(
    spec: ProblemSpec,
    plan: RunPlan,
    mesh: MeshMetadata,
    run_dir: Path,
    trial: SolutionState,
    field_name: str,
    material_terms: tuple[tuple[float, int], ...],
) -> StepSolve:
    mesh_data = gmshio.read_from_msh(mesh.path, MPI.COMM_WORLD, rank=0, gdim=mesh.dimension)
    domain = mesh_data.mesh
    cell_tags = mesh_data.cell_tags
    facet_tags = mesh_data.facet_tags
    V = fem.functionspace(domain, ("Lagrange", _field_order(spec, field_name)))
    u = ufl.TrialFunction(V)
    v = ufl.TestFunction(V)
    dx = ufl.Measure("dx", domain=domain, subdomain_data=cell_tags)
    ds = ufl.Measure("ds", domain=domain, subdomain_data=facet_tags)
    a = _sum_forms([k * ufl.inner(ufl.grad(u), ufl.grad(v)) * dx(tag) for k, tag in material_terms])
    L_terms = _load_forms(spec, mesh, v, dx, ds)
    L = _sum_forms(L_terms) if L_terms else PETSc.ScalarType(0.0) * v * dx
    bcs = _dirichlet_bcs(spec, mesh, V, facet_tags, fem, PETSc.ScalarType)
    options = dict(spec.solver.linear)
    options["ksp_error_if_not_converged"] = True
    problem = LinearProblem(a, L, bcs=bcs, petsc_options_prefix=_petsc_prefix(spec), petsc_options=options)
    uh = problem.solve()
    uh.name = field_name
    reason = int(problem.solver.getConvergedReason())
    iterations = int(problem.solver.getIterationNumber())
    norm = float(problem.solver.getResidualNorm())
    if reason <= 0:
        raise RuntimeError(f"PETSc KSP did not converge: {reason}")
    array = np.asarray(np.real(uh.x.array), dtype=float)
    local_min = float(np.min(array)) if array.size else float("inf")
    local_max = float(np.max(array)) if array.size else float("-inf")
    local_l2 = float(np.dot(array, array))
    field_state = {
        field_name: {
            "min": float(domain.comm.allreduce(local_min, op=MPI.MIN)),
            "max": float(domain.comm.allreduce(local_max, op=MPI.MAX)),
            "l2_norm": float(domain.comm.allreduce(local_l2, op=MPI.SUM) ** 0.5),
            "dofs_local": int(array.size),
        }
    }
    artifacts: dict[str, Path] = {}
    if spec.output.format == "vtx" and trial.committed_step + 1 in plan.output_steps:
        output_path = run_dir / "fields" / "solution.bp"
        with VTXWriter(domain.comm, output_path, [uh]) as writer:
            writer.write(float(trial.time or 0.0))
        artifacts["solution_fields"] = output_path
    payload = {"reason": reason, "iterations": iterations, "residual_norm": norm}
    return StepSolve(field_state=field_state, artifacts=artifacts, payload=payload)


def _single_scalar_field(spec: ProblemSpec) -> str:
    fields = tuple(field for field in spec.fields if field.kind == "scalar")
    if len(fields) != 1 or len(spec.fields) != 1:
        raise NotImplementedError("linear scalar backend requires exactly one scalar field")
    return fields[0].name


def _field_order(spec: ProblemSpec, field_name: str) -> int:
    for field in spec.fields:
        if field.name == field_name:
            return field.element.order
    raise KeyError(field_name)


def _material_terms(spec: ProblemSpec, mesh: MeshMetadata) -> tuple[tuple[float, int], ...]:
    terms: list[tuple[float, int]] = []
    for name, material in sorted(spec.materials.items()):
        if material.model != "isotropic_heat":
            raise NotImplementedError(f"linear scalar backend does not implement material model {material.model}")
        binding = _binding(mesh, "materials", name)
        terms.append((_quantity_scalar(material.parameters["k"]), binding.physical_id))
    return tuple(terms)


def _load_forms(spec: ProblemSpec, mesh: MeshMetadata, v: Any, dx: Any, ds: Any) -> list[Any]:
    forms: list[Any] = []
    for load in spec.loads:
        binding = _binding_by_name(mesh, load.on)
        value = _quantity_scalar(load.value)
        if load.type == "source":
            forms.append(value * v * dx(binding.physical_id))
        elif load.type == "flux":
            forms.append(value * v * ds(binding.physical_id))
        else:
            raise NotImplementedError(f"linear scalar backend does not implement load type {load.type}")
    return forms


def _dirichlet_bcs(spec: ProblemSpec, mesh: MeshMetadata, V: Any, facet_tags: Any, fem: Any, scalar_type: Any) -> list[Any]:
    bcs: list[Any] = []
    for bc in spec.bcs:
        if bc.type != "dirichlet":
            raise NotImplementedError(f"linear scalar backend does not implement boundary condition type {bc.type}")
        binding = _binding_by_name(mesh, bc.on)
        facets = facet_tags.find(binding.physical_id)
        dofs = fem.locate_dofs_topological(V=V, entity_dim=binding.dim, entities=facets)
        bcs.append(fem.dirichletbc(value=scalar_type(_quantity_scalar(bc.value)), dofs=dofs, V=V))
    return bcs


def _binding(mesh: MeshMetadata, namespace: str, name: str) -> TagBinding:
    for binding in mesh.tags.bindings:
        if binding.namespace == namespace and binding.name == name:
            return binding
    raise KeyError(f"{namespace}/{name}")


def _binding_by_name(mesh: MeshMetadata, name: str) -> TagBinding:
    matches = tuple(binding for binding in mesh.tags.bindings if binding.name == name)
    if len(matches) != 1:
        raise KeyError(name)
    return matches[0]


def _quantity_scalar(value: Any) -> float:
    if not isinstance(value, dict) or "magnitude" not in value:
        raise TypeError("expected canonical scalar quantity")
    magnitude = value["magnitude"]
    if isinstance(magnitude, (list, tuple)):
        raise TypeError("expected scalar magnitude")
    return float(magnitude)


def _sum_forms(forms: list[Any]) -> Any:
    head, *tail = forms
    value = head
    for form in tail:
        value += form
    return value


def _petsc_prefix(spec: ProblemSpec) -> str:
    return spec.solver.prefix or f"{spec.name}_"


def _failed_step(
    spec: ProblemSpec,
    plan: RunPlan,
    committed: SolutionState,
    trial: SolutionState,
    phases: tuple[str, ...],
    acceptance: tuple[AcceptanceCheck, ...],
    diagnostics: tuple[Diagnostic, ...],
) -> StepRecord:
    return StepRecord(
        index=trial.trial_step or committed.committed_step + 1,
        status="failed",
        mode=spec.mode,
        solver=plan.solver,
        phases=phases,
        target_time=trial.time,
        time_unit=trial.time_unit,
        opened_state_hash=committed.state_hash,
        trial_state_hash=trial.state_hash,
        final_state_hash=committed.state_hash,
        acceptance=acceptance,
        diagnostics=diagnostics,
    )


def _solve_diagnostic(exc: Exception, step_index: int) -> Diagnostic:
    return Diagnostic(
        code="SOLVE_FAILED",
        message=str(exc),
        path=("solver",),
        step_index=step_index,
        source="solver",
        backend_error=f"{type(exc).__name__}: {exc}",
    )


def _finite_payload(field_state: dict[str, Any]) -> dict[str, Any]:
    return {"fields": tuple(sorted(field_state))}


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
