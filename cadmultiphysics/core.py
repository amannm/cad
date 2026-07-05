from __future__ import annotations

import hashlib
import importlib.metadata
import json
import os
import platform
from typing import Any

from pydantic import ValidationError

from cadmultiphysics.diagnostics import Diagnostic, schema_diagnostics
from cadmultiphysics.errors import SchemaError, UnitError
from cadmultiphysics.schema import (
    BoundaryConditionSpec,
    DomainIR,
    ElementSpec,
    FieldSpec,
    GeometryEntityInput,
    GeometryEntitySpec,
    GeometrySpec,
    LoadSpec,
    MaterialSpec,
    MeshPlan,
    MeshSizeSpec,
    OutputPlan,
    PETScProfile,
    ProblemInput,
    ProblemSpec,
    RunManifest,
    RunPlan,
    SemanticTagInput,
    SemanticTagSpec,
    TagsSpec,
    TimeGrid,
    TimePlan,
)
from cadmultiphysics.units import UnitDiagnostics, canonical_quantity, canonical_value, unit_spec

LENGTH = "[length]"
TIME = "[time]"
DIMENSIONLESS = "dimensionless"


def build_problem_spec(data: dict[str, Any]) -> ProblemSpec:
    try:
        raw = ProblemInput.model_validate(data)
    except ValidationError as exc:
        raise SchemaError(schema_diagnostics(exc)) from exc
    diagnostics = _semantic_diagnostics(raw)
    if diagnostics:
        raise SchemaError(diagnostics)
    try:
        spec = _canonical_problem(raw)
    except UnitDiagnostics as exc:
        raise UnitError(exc.diagnostics) from exc
    return spec.model_copy(update={"content_hash": content_hash(spec)})


def build_domain_ir(spec: ProblemSpec) -> DomainIR:
    return DomainIR(
        name=spec.name,
        mode=spec.mode,
        entities=tuple(entity.name for entity in spec.geometry.entities),
        material_tags=tuple(sorted(spec.tags.materials)),
        boundary_tags=tuple(sorted((*spec.tags.boundaries, *spec.tags.interfaces, *spec.tags.curves, *spec.tags.points))),
        fields=tuple(field.name for field in spec.fields),
        bcs=tuple(bc.name for bc in spec.bcs),
        loads=tuple(load.name for load in spec.loads),
        content_hash=spec.content_hash,
    )


def build_run_manifest(spec: ProblemSpec, run_dir: str) -> RunManifest:
    return RunManifest(
        schema_version=spec.version,
        content_hash=spec.content_hash,
        backend_versions={
            "cadmultiphysics": _version("cadmultiphysics"),
            "pydantic": _version("pydantic"),
            "pint": _version("pint"),
            "gmsh": _version("gmsh"),
            "dolfinx": _version("fenics-dolfinx"),
            "ufl": _version("fenics-ufl"),
            "petsc4py": _version("petsc4py"),
            "adios2": _version("adios2"),
        },
        python_version=platform.python_version(),
        mpi_size=int(os.environ.get("OMPI_COMM_WORLD_SIZE") or os.environ.get("PMI_SIZE") or "1"),
        mesh_options=spec.mesh.model_dump(mode="json"),
        output_paths={
            "run_dir": run_dir,
            "manifest": f"{run_dir}/manifest.json",
            "report": f"{run_dir}/report.json",
            "mesh": f"{run_dir}/mesh",
            "fields": f"{run_dir}/fields",
            "restarts": f"{run_dir}/restarts",
            "logs": f"{run_dir}/logs",
        },
        restart={
            "schema_version": spec.version,
            "manifest_hash_required": True,
            "field_layout": tuple(field.name for field in spec.fields),
        },
    )


def build_run_plan(spec: ProblemSpec) -> RunPlan:
    linear = spec.mode.startswith("linear")
    transient = spec.mode.endswith("_transient")
    time = _time_grid(spec) if transient else None
    return RunPlan(
        mode=spec.mode,
        problem_kind="linear" if linear else "nonlinear",
        transient=transient,
        solver="ksp" if linear else "snes",
        steps=time.steps if time else 1,
        time=time,
        output_cadence=spec.output.cadence,
        restart_cadence=spec.output.restart_cadence,
    )


def content_hash(spec: ProblemSpec) -> str:
    payload = spec.model_dump(mode="json", exclude={"content_hash"})
    encoded = json.dumps(payload, sort_keys=True, separators=(",", ":"), ensure_ascii=True).encode()
    return hashlib.sha256(encoded).hexdigest()


def input_json_schema() -> dict[str, Any]:
    return ProblemInput.model_json_schema()


def _semantic_diagnostics(raw: ProblemInput) -> list[Diagnostic]:
    diagnostics: list[Diagnostic] = []
    entity_names = [entity.name for entity in raw.geometry.entities]
    field_names = [field.name for field in raw.fields]
    material_names = set(raw.materials)
    all_tags = _all_tags(raw.tags)
    boundary_like_tags = {
        *raw.tags.boundaries,
        *raw.tags.interfaces,
        *raw.tags.curves,
        *raw.tags.points,
    }
    _duplicates(entity_names, ("geometry", "entities"), "GEOMETRY_DUPLICATE_ENTITY", diagnostics)
    _duplicates(field_names, ("fields",), "FIELD_DUPLICATE", diagnostics)
    _duplicates([bc.name for bc in raw.boundary_conditions], ("boundary_conditions",), "BC_DUPLICATE", diagnostics)
    _duplicates([load.name for load in raw.loads], ("loads",), "LOAD_DUPLICATE", diagnostics)
    entity_set = set(entity_names)
    field_set = set(field_names)
    for index, entity in enumerate(raw.geometry.entities):
        diagnostics.extend(_entity_diagnostics(entity, ("geometry", "entities", index)))
    for namespace, tags, expected_dim in (
        ("materials", raw.tags.materials, 3),
        ("boundaries", raw.tags.boundaries, 2),
        ("interfaces", raw.tags.interfaces, 2),
        ("curves", raw.tags.curves, 1),
        ("points", raw.tags.points, 0),
    ):
        for name, tag in tags.items():
            path = ("tags", namespace, name)
            if tag.dim != expected_dim:
                diagnostics.append(
                    Diagnostic(
                        code="TAG_DIMENSION_INVALID",
                        message=f"Tag '{name}' must have dimension {expected_dim}.",
                        path=path,
                        source="tags",
                    )
                )
            diagnostics.extend(_tag_entity_diagnostics(tag, entity_set, path))
    for material in material_names:
        if material not in raw.tags.materials:
            diagnostics.append(
                Diagnostic(
                    code="MATERIAL_TAG_MISSING",
                    message=f"Material '{material}' has no material tag.",
                    path=("materials", material),
                    source="schema",
                )
            )
    for tag in raw.tags.materials:
        if tag not in material_names:
            diagnostics.append(
                Diagnostic(
                    code="MATERIAL_MODEL_MISSING",
                    message=f"Material tag '{tag}' has no material model.",
                    path=("tags", "materials", tag),
                    source="schema",
                )
            )
    for index, field in enumerate(raw.fields):
        if field.kind == "scalar" and field.components not in (None, 1):
            diagnostics.append(
                Diagnostic(
                    code="FIELD_COMPONENTS_INVALID",
                    message=f"Scalar field '{field.name}' must have one component.",
                    path=("fields", index, "components"),
                    source="schema",
                )
            )
        if field.kind == "vector" and field.components is None:
            diagnostics.append(
                Diagnostic(
                    code="FIELD_COMPONENTS_REQUIRED",
                    message=f"Vector field '{field.name}' requires components.",
                    path=("fields", index, "components"),
                    source="schema",
                )
            )
    for index, bc in enumerate(raw.boundary_conditions):
        path = ("boundary_conditions", index)
        if bc.field not in field_set:
            diagnostics.append(_unknown("BC_FIELD_UNKNOWN", f"Boundary condition '{bc.name}' references unknown field '{bc.field}'.", (*path, "field")))
        if bc.on not in boundary_like_tags:
            diagnostics.append(_unknown("BC_TAG_UNKNOWN", f"Boundary condition '{bc.name}' references unknown boundary tag '{bc.on}'.", (*path, "on")))
    for index, load in enumerate(raw.loads):
        path = ("loads", index)
        if load.field not in field_set:
            diagnostics.append(_unknown("LOAD_FIELD_UNKNOWN", f"Load '{load.name}' references unknown field '{load.field}'.", (*path, "field")))
        if load.on not in all_tags:
            diagnostics.append(_unknown("LOAD_TAG_UNKNOWN", f"Load '{load.name}' references unknown tag '{load.on}'.", (*path, "on")))
    for index, name in enumerate(raw.output.fields):
        if name not in field_set:
            diagnostics.append(_unknown("OUTPUT_FIELD_UNKNOWN", f"Output references unknown field '{name}'.", ("output", "fields", index)))
    for name in raw.solver.fieldsplits:
        if name not in field_set:
            diagnostics.append(_unknown("SOLVER_FIELDSPLIT_UNKNOWN", f"Fieldsplit '{name}' does not match a declared field.", ("solver", "fieldsplits", name)))
    if raw.mode.endswith("_transient") and raw.time is None:
        diagnostics.append(
            Diagnostic(
                code="TIME_REQUIRED",
                message=f"Mode '{raw.mode}' requires a time block.",
                path=("time",),
                source="schema",
            )
        )
    if raw.mode.endswith("_steady") and raw.time is not None:
        diagnostics.append(
            Diagnostic(
                code="TIME_UNEXPECTED",
                message=f"Mode '{raw.mode}' must not include a time block.",
                path=("time",),
                source="schema",
            )
        )
    if raw.solver.prefix and not raw.solver.prefix.replace("_", "").isalnum():
        diagnostics.append(
            Diagnostic(
                code="SOLVER_PREFIX_INVALID",
                message="Solver prefix may contain only letters, numbers, and underscores.",
                path=("solver", "prefix"),
                source="solver",
            )
        )
    return diagnostics


def _canonical_problem(raw: ProblemInput) -> ProblemSpec:
    fields = tuple(_field_spec(field) for field in raw.fields)
    field_dimensions = {field.name: field.unit.dimension for field in fields}
    field_components = {field.name: field.components for field in fields}
    spec = ProblemSpec(
        name=raw.name,
        version=raw.version,
        mode=raw.mode,
        geometry=GeometrySpec(
            backend=raw.geometry.backend,
            entities=tuple(_entity_spec(entity, index) for index, entity in enumerate(raw.geometry.entities)),
        ),
        tags=_tags_spec(raw),
        fields=fields,
        materials={
            name: MaterialSpec(
                model=material.model,
                parameters={key: canonical_value(value, ("materials", name, "parameters", key)) for key, value in sorted(material.parameters.items())},
            )
            for name, material in sorted(raw.materials.items())
        },
        bcs=tuple(_bc_spec(bc, index, field_dimensions, field_components) for index, bc in enumerate(raw.boundary_conditions)),
        loads=tuple(_load_spec(load, index) for index, load in enumerate(raw.loads)),
        mesh=_mesh_plan(raw.mesh),
        solver=PETScProfile(
            prefix=raw.solver.prefix,
            linear=dict(sorted(raw.solver.linear.items())),
            nonlinear=dict(sorted(raw.solver.nonlinear.items())),
            fieldsplits={name: dict(sorted(options.items())) for name, options in sorted(raw.solver.fieldsplits.items())},
            allow_backend_options=raw.solver.allow_backend_options,
        ),
        output=OutputPlan(
            format=raw.output.format,
            fields=raw.output.fields,
            derived_fields=raw.output.derived_fields,
            cadence=raw.output.cadence,
            restart_cadence=raw.output.restart_cadence,
            report_formats=raw.output.report_formats,
            writer_options=dict(sorted(raw.output.writer_options.items())),
        ),
        time=_time_plan(raw.time) if raw.time else None,
        metadata=dict(sorted(raw.metadata.items())),
    )
    _validate_time_values(spec)
    return spec


def _entity_spec(entity: GeometryEntityInput, index: int) -> GeometryEntitySpec:
    path = ("geometry", "entities", index)
    if entity.type == "box":
        return GeometryEntitySpec(
            type=entity.type,
            name=entity.name,
            size=canonical_quantity(entity.size, (*path, "size"), LENGTH),
            origin=canonical_quantity(entity.origin, (*path, "origin"), LENGTH) if entity.origin is not None else None,
        )
    return GeometryEntitySpec(
        type=entity.type,
        name=entity.name,
        radius=canonical_quantity(entity.radius, (*path, "radius"), LENGTH),
        height=canonical_quantity(entity.height, (*path, "height"), LENGTH),
        axis=_axis(entity.axis) if entity.axis is not None else None,
    )


def _field_spec(field: Any) -> FieldSpec:
    return FieldSpec(
        name=field.name,
        kind=field.kind,
        components=1 if field.components is None else field.components,
        unit=unit_spec(field.unit, ("fields", field.name, "unit")),
        element=ElementSpec(family=field.element.family, order=field.element.order),
    )


def _tags_spec(raw: ProblemInput) -> TagsSpec:
    return TagsSpec(
        materials={name: _tag_spec(tag) for name, tag in sorted(raw.tags.materials.items())},
        boundaries={name: _tag_spec(tag) for name, tag in sorted(raw.tags.boundaries.items())},
        interfaces={name: _tag_spec(tag) for name, tag in sorted(raw.tags.interfaces.items())},
        curves={name: _tag_spec(tag) for name, tag in sorted(raw.tags.curves.items())},
        points={name: _tag_spec(tag) for name, tag in sorted(raw.tags.points.items())},
    )


def _tag_spec(tag: SemanticTagInput) -> SemanticTagSpec:
    return SemanticTagSpec(dim=tag.dim, entities=tuple(sorted(tag.entities)), selector=tag.selector.strip() if tag.selector else None)


def _bc_spec(
    bc: Any,
    index: int,
    field_dimensions: dict[str, str],
    field_components: dict[str, int],
) -> BoundaryConditionSpec:
    value = None
    if bc.value is not None:
        value = canonical_quantity(bc.value, ("boundary_conditions", index, "value"), field_dimensions[bc.field]).model_dump(mode="json")
        magnitude = value["magnitude"]
        if (isinstance(magnitude, list) and len(magnitude) != field_components[bc.field]) or (
            not isinstance(magnitude, list) and field_components[bc.field] != 1
        ):
            raise UnitDiagnostics(
                [
                    Diagnostic(
                        code="BC_VALUE_COMPONENT_MISMATCH",
                        message=f"Boundary condition '{bc.name}' expects {field_components[bc.field]} components.",
                        path=("boundary_conditions", index, "value"),
                        source="schema",
                    )
                ]
            )
    return BoundaryConditionSpec(
        name=bc.name,
        on=bc.on,
        field=bc.field,
        type=bc.type,
        value=value,
        parameters={key: canonical_value(value, ("boundary_conditions", index, "parameters", key)) for key, value in sorted(bc.parameters.items())},
    )


def _load_spec(load: Any, index: int) -> LoadSpec:
    return LoadSpec(
        name=load.name,
        on=load.on,
        field=load.field,
        type=load.type,
        value=canonical_value(load.value, ("loads", index, "value")) if load.value is not None else None,
        parameters={key: canonical_value(value, ("loads", index, "parameters", key)) for key, value in sorted(load.parameters.items())},
    )


def _mesh_plan(mesh: Any) -> MeshPlan:
    return MeshPlan(
        cell_type=mesh.cell_type,
        dimension=mesh.dimension or _cell_dimension(mesh.cell_type),
        order=mesh.order,
        size=MeshSizeSpec(
            global_size=canonical_quantity(mesh.size.global_size, ("mesh", "size", "global"), LENGTH),
            local={name: canonical_quantity(value, ("mesh", "size", "local", name), LENGTH) for name, value in sorted(mesh.size.local.items())},
        ),
        curvature=mesh.curvature,
        partitions=mesh.partitions,
        quality={key: canonical_value(value, ("mesh", "quality", key)) for key, value in sorted(mesh.quality.items())},
    )


def _time_plan(time: Any) -> TimePlan:
    return TimePlan(
        start=canonical_quantity(time.start, ("time", "start"), TIME),
        stop=canonical_quantity(time.stop, ("time", "stop"), TIME),
        step=canonical_quantity(time.step, ("time", "step"), TIME),
        scheme=time.scheme,
    )


def _validate_time_values(spec: ProblemSpec) -> None:
    if spec.time is None:
        return
    if not isinstance(spec.time.start.magnitude, float) or not isinstance(spec.time.stop.magnitude, float) or not isinstance(spec.time.step.magnitude, float):
        raise UnitDiagnostics(
            [
                Diagnostic(
                    code="TIME_SCALAR_REQUIRED",
                    message="Time start, stop, and step must be scalar quantities.",
                    path=("time",),
                    source="schema",
                )
            ]
        )
    if spec.time.stop.magnitude <= spec.time.start.magnitude:
        raise UnitDiagnostics(
            [
                Diagnostic(
                    code="TIME_RANGE_INVALID",
                    message="Time stop must be greater than start.",
                    path=("time", "stop"),
                    source="schema",
                )
            ]
        )
    if spec.time.step.magnitude <= 0:
        raise UnitDiagnostics(
            [
                Diagnostic(
                    code="TIME_STEP_INVALID",
                    message="Time step must be positive.",
                    path=("time", "step"),
                    source="schema",
                )
            ]
        )


def _time_grid(spec: ProblemSpec) -> TimeGrid:
    if spec.time is None:
        raise UnitDiagnostics(
            [
                Diagnostic(
                    code="TIME_REQUIRED",
                    message=f"Mode '{spec.mode}' requires a time block.",
                    path=("time",),
                    source="schema",
                )
            ]
        )
    start = float(spec.time.start.magnitude)
    stop = float(spec.time.stop.magnitude)
    step = float(spec.time.step.magnitude)
    span = stop - start
    steps = int(span // step)
    if start + steps * step < stop:
        steps += 1
    return TimeGrid(start=start, stop=stop, step=step, steps=steps, unit=spec.time.step.unit)


def _entity_diagnostics(entity: GeometryEntityInput, path: tuple[str | int, ...]) -> list[Diagnostic]:
    diagnostics: list[Diagnostic] = []
    if entity.type == "box":
        if entity.size is None or len(entity.size) != 3:
            diagnostics.append(Diagnostic(code="GEOMETRY_BOX_SIZE_INVALID", message=f"Box '{entity.name}' requires a 3-vector size.", path=(*path, "size"), source="geometry"))
        if entity.origin is not None and len(entity.origin) != 3:
            diagnostics.append(Diagnostic(code="GEOMETRY_BOX_ORIGIN_INVALID", message=f"Box '{entity.name}' origin must be a 3-vector.", path=(*path, "origin"), source="geometry"))
        if entity.radius is not None or entity.height is not None or entity.axis is not None:
            diagnostics.append(Diagnostic(code="GEOMETRY_FIELD_INVALID", message=f"Box '{entity.name}' contains cylinder-only fields.", path=path, source="geometry"))
    if entity.type == "cylinder":
        if entity.radius is None:
            diagnostics.append(Diagnostic(code="GEOMETRY_CYLINDER_RADIUS_REQUIRED", message=f"Cylinder '{entity.name}' requires radius.", path=(*path, "radius"), source="geometry"))
        if entity.height is None:
            diagnostics.append(Diagnostic(code="GEOMETRY_CYLINDER_HEIGHT_REQUIRED", message=f"Cylinder '{entity.name}' requires height.", path=(*path, "height"), source="geometry"))
        if entity.axis is not None and len(entity.axis) != 3:
            diagnostics.append(Diagnostic(code="GEOMETRY_CYLINDER_AXIS_INVALID", message=f"Cylinder '{entity.name}' axis must be a 3-vector.", path=(*path, "axis"), source="geometry"))
        if entity.size is not None:
            diagnostics.append(Diagnostic(code="GEOMETRY_FIELD_INVALID", message=f"Cylinder '{entity.name}' contains box-only fields.", path=path, source="geometry"))
    return diagnostics


def _tag_entity_diagnostics(tag: SemanticTagInput, entity_set: set[str], path: tuple[str | int, ...]) -> list[Diagnostic]:
    return [
        Diagnostic(
            code="TAG_ENTITY_UNKNOWN",
            message=f"Tag references unknown geometry entity '{entity}'.",
            path=(*path, "entities", index),
            source="tags",
        )
        for index, entity in enumerate(tag.entities)
        if entity not in entity_set
    ]


def _duplicates(values: list[str], path: tuple[str | int, ...], code: str, diagnostics: list[Diagnostic]) -> None:
    seen: set[str] = set()
    for index, value in enumerate(values):
        if value in seen:
            diagnostics.append(
                Diagnostic(
                    code=code,
                    message=f"Duplicate name '{value}'.",
                    path=(*path, index),
                    source="schema",
                )
            )
        seen.add(value)


def _all_tags(tags: Any) -> set[str]:
    return {*tags.materials, *tags.boundaries, *tags.interfaces, *tags.curves, *tags.points}


def _unknown(code: str, message: str, path: tuple[str | int, ...]) -> Diagnostic:
    return Diagnostic(code=code, message=message, path=path, source="schema")


def _cell_dimension(cell_type: str) -> int:
    return {"tetrahedron": 3, "hexahedron": 3, "triangle": 2, "quadrilateral": 2, "interval": 1}[cell_type]


def _axis(axis: tuple[Any, ...]) -> tuple[float, float, float]:
    if not all(isinstance(item, int | float) for item in axis):
        raise UnitDiagnostics(
            [
                Diagnostic(
                    code="GEOMETRY_AXIS_INVALID",
                    message="Cylinder axis must be numeric and dimensionless.",
                    path=("geometry", "axis"),
                    source="geometry",
                )
            ]
        )
    return (float(axis[0]), float(axis[1]), float(axis[2]))


def _version(package: str) -> str | None:
    try:
        return importlib.metadata.version(package)
    except importlib.metadata.PackageNotFoundError:
        return None
