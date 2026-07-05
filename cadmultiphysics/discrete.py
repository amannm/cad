from __future__ import annotations

import hashlib
import json
from collections.abc import Iterable
from typing import Any

from cadmultiphysics.diagnostics import Diagnostic
from cadmultiphysics.errors import DiscretizationError
from cadmultiphysics.schema import (
    BoundaryConditionBindingPlan,
    CouplingDependencyPlan,
    DiscretePlan,
    FieldSpec,
    FunctionSpacePlan,
    LoadBindingPlan,
    MaterialBindingPlan,
    MeasureBindingPlan,
    MeshMetadata,
    ProblemSpec,
    TagBinding,
)
from cadmultiphysics.units import dimension_of

TEMPERATURE = dimension_of("K")
LENGTH = "[length]"
_BOUNDARY_NAMESPACES = {"boundaries", "interfaces", "curves", "points"}


def build_discrete_plan(spec: ProblemSpec, mesh: MeshMetadata) -> DiscretePlan:
    diagnostics: list[Diagnostic] = []
    by_key = {(binding.namespace, binding.name): binding for binding in mesh.tags.bindings}
    boundary_by_name = _by_name(binding for binding in mesh.tags.bindings if binding.namespace in _BOUNDARY_NAMESPACES)
    all_by_name = _by_name(mesh.tags.bindings)
    spaces = tuple(
        FunctionSpacePlan(
            field=field.name,
            kind=field.kind,
            components=field.components,
            element_family=field.element.family,
            element_order=field.element.order,
            unit=field.unit.unit,
            unit_dimension=field.unit.dimension,
            block_index=index,
        )
        for index, field in enumerate(spec.fields)
    )
    measures = tuple(
        MeasureBindingPlan(
            namespace=binding.namespace,
            name=binding.name,
            dim=binding.dim,
            physical_id=binding.physical_id,
            physical_name=binding.physical_name,
        )
        for binding in sorted(mesh.tags.bindings, key=lambda item: item.physical_id)
    )
    materials: list[MaterialBindingPlan] = []
    for name, material in sorted(spec.materials.items()):
        binding = by_key.get(("materials", name))
        if binding is None:
            diagnostics.append(
                Diagnostic(
                    code="DISCRETE_MATERIAL_TAG_MISSING",
                    message=f"Material '{name}' has no physical material tag.",
                    path=("materials", name),
                    source="discrete",
                )
            )
            continue
        materials.append(
            MaterialBindingPlan(
                material=name,
                model=material.model,
                tag=name,
                dim=binding.dim,
                physical_id=binding.physical_id,
                entity_tags=binding.entity_tags,
            )
        )
    bcs: list[BoundaryConditionBindingPlan] = []
    for index, bc in enumerate(spec.bcs):
        binding = _resolve_tag(bc.on, boundary_by_name, ("boundary_conditions", index, "on"), diagnostics)
        if binding is None:
            continue
        bcs.append(
            BoundaryConditionBindingPlan(
                name=bc.name,
                type=bc.type,
                field=bc.field,
                tag=bc.on,
                dim=binding.dim,
                physical_id=binding.physical_id,
                value=bc.value,
                parameters=dict(sorted(bc.parameters.items())),
            )
        )
    loads: list[LoadBindingPlan] = []
    for index, load in enumerate(spec.loads):
        path = ("loads", index)
        binding = _resolve_tag(load.on, all_by_name, (*path, "on"), diagnostics)
        if binding is None:
            continue
        diagnostics.extend(_load_dimension_diagnostics(load.type, binding, path))
        loads.append(
            LoadBindingPlan(
                name=load.name,
                type=load.type,
                field=load.field,
                tag=load.on,
                dim=binding.dim,
                physical_id=binding.physical_id,
                value=load.value,
                parameters=dict(sorted(load.parameters.items())),
            )
        )
    if diagnostics:
        raise DiscretizationError(diagnostics)
    plan = DiscretePlan(
        spec_hash=spec.content_hash,
        mesh_hash=_mesh_hash(mesh),
        mode=spec.mode,
        mesh_dimension=mesh.dimension,
        cell_type=mesh.cell_type,
        spaces=spaces,
        measures=measures,
        materials=tuple(materials),
        boundary_conditions=tuple(bcs),
        loads=tuple(loads),
        coupling=_coupling(spec.fields, tuple(material.model for material in spec.materials.values())),
        solver_fieldsplits=tuple(field.name for field in spec.fields if field.name in spec.solver.fieldsplits),
        output_fields=spec.output.fields,
    )
    return plan.model_copy(update={"content_hash": _hash_payload(plan.model_dump(mode="json", exclude={"content_hash"}))})


def _by_name(bindings: Iterable[TagBinding]) -> dict[str, tuple[TagBinding, ...]]:
    result: dict[str, list[TagBinding]] = {}
    for binding in bindings:
        result.setdefault(binding.name, []).append(binding)
    return {name: tuple(sorted(values, key=lambda item: item.physical_name)) for name, values in sorted(result.items())}


def _resolve_tag(
    name: str,
    index: dict[str, tuple[TagBinding, ...]],
    path: tuple[str | int, ...],
    diagnostics: list[Diagnostic],
) -> TagBinding | None:
    matches = index.get(name, ())
    if len(matches) == 1:
        return matches[0]
    if not matches:
        diagnostics.append(
            Diagnostic(
                code="DISCRETE_TAG_MISSING",
                message=f"Semantic tag '{name}' was not emitted as a physical group.",
                path=path,
                source="discrete",
            )
        )
        return None
    diagnostics.append(
        Diagnostic(
            code="DISCRETE_TAG_AMBIGUOUS",
            message=f"Semantic tag reference '{name}' matches multiple physical groups.",
            path=path,
            source="discrete",
            payload={"matches": tuple(binding.physical_name for binding in matches)},
        )
    )
    return None


def _load_dimension_diagnostics(load_type: str, binding: TagBinding, path: tuple[str | int, ...]) -> list[Diagnostic]:
    if load_type in {"flux", "traction"} and binding.dim != 2:
        return [
            Diagnostic(
                code="DISCRETE_LOAD_DIMENSION_INVALID",
                message=f"Surface load '{load_type}' must target a facet tag.",
                path=(*path, "on"),
                source="discrete",
                payload={"tag": binding.physical_name, "dim": binding.dim},
            )
        ]
    if load_type in {"source", "body_force"} and binding.dim != 3:
        return [
            Diagnostic(
                code="DISCRETE_LOAD_DIMENSION_INVALID",
                message=f"Domain load '{load_type}' must target a cell tag.",
                path=(*path, "on"),
                source="discrete",
                payload={"tag": binding.physical_name, "dim": binding.dim},
            )
        ]
    return []


def _coupling(fields: tuple[FieldSpec, ...], material_models: tuple[str, ...]) -> tuple[CouplingDependencyPlan, ...]:
    if "thermoelastic_small_strain" not in material_models:
        return ()
    temperatures = tuple(field.name for field in fields if field.kind == "scalar" and field.unit.dimension == TEMPERATURE)
    displacements = tuple(field.name for field in fields if field.kind == "vector" and field.unit.dimension == LENGTH)
    return tuple(
        CouplingDependencyPlan(source_field=temperature, target_field=displacement, mechanism="thermal_strain")
        for temperature in temperatures
        for displacement in displacements
    )


def _mesh_hash(mesh: MeshMetadata) -> str:
    return _hash_payload(mesh.model_dump(mode="json", exclude={"path"}))


def _hash_payload(payload: dict[str, Any]) -> str:
    encoded = json.dumps(payload, sort_keys=True, separators=(",", ":"), ensure_ascii=True).encode()
    return hashlib.sha256(encoded).hexdigest()
