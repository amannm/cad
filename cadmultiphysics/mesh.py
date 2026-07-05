from __future__ import annotations

import math
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import gmsh

from cadmultiphysics.diagnostics import Diagnostic
from cadmultiphysics.errors import MeshError
from cadmultiphysics.schema import (
    GeometryEntityIR,
    MeshMetadata,
    ProblemSpec,
    SemanticTagSpec,
    TagBinding,
    TagMap,
    TagNamespace,
)
from cadmultiphysics.units import canonical_quantity

_SELECTOR = re.compile(r"^\s*([xyz])\s*(==|<=|>=|<|>)\s*(.+?)\s*$")
_NUMBER = re.compile(r"^[+-]?(?:\d+(?:\.\d*)?|\.\d+)(?:[eE][+-]?\d+)?$")
_NAMESPACES: tuple[TagNamespace, ...] = ("materials", "boundaries", "interfaces", "curves", "points")


@dataclass(frozen=True)
class MeshBuild:
    metadata: MeshMetadata
    artifacts: dict[str, Path]


def generate_mesh(spec: ProblemSpec, mesh_dir: Path) -> MeshBuild:
    diagnostics = _cell_type_diagnostics(spec)
    if diagnostics:
        raise MeshError(diagnostics)
    mesh_dir.mkdir(parents=True, exist_ok=True)
    gmsh.initialize()
    try:
        gmsh.option.setNumber("General.Terminal", 0)
        gmsh.option.setNumber("Mesh.MshFileVersion", 4.1)
        gmsh.model.add(spec.name)
        entities = _build_geometry(spec)
        gmsh.model.occ.synchronize()
        entities = {name: entity.model_copy(update={"bounds": _bounds(entity.dim, entity.backend_tag)}) for name, entity in entities.items()}
        tags = _build_tag_map(spec, entities)
        for binding in tags.bindings:
            gmsh.model.addPhysicalGroup(binding.dim, list(binding.entity_tags), binding.physical_id)
            gmsh.model.setPhysicalName(binding.dim, binding.physical_id, binding.physical_name)
        _apply_sizes(spec, entities, tags)
        gmsh.model.mesh.generate(spec.mesh.dimension)
        gmsh.model.mesh.setOrder(spec.mesh.order)
        mesh_path = mesh_dir / "model.msh"
        gmsh.write(str(mesh_path))
        metadata = _metadata(spec, mesh_path, entities, tags)
        tag_path = mesh_dir / "tag_map.json"
        metadata_path = mesh_dir / "mesh_metadata.json"
        geometry_path = mesh_dir / "geometry_ir.json"
        return MeshBuild(
            metadata=metadata,
            artifacts={
                "model_msh": mesh_path,
                "mesh_metadata": metadata_path,
                "tag_map": tag_path,
                "geometry_ir": geometry_path,
            },
        )
    except MeshError:
        raise
    except Exception as exc:
        raise MeshError(
            [
                Diagnostic(
                    code="MESH_GENERATION_FAILED",
                    message="Gmsh mesh generation failed.",
                    path=("mesh",),
                    source="mesh",
                    backend_error=str(exc),
                )
            ]
        ) from exc
    finally:
        gmsh.finalize()


def _build_geometry(spec: ProblemSpec) -> dict[str, GeometryEntityIR]:
    entities: dict[str, GeometryEntityIR] = {}
    for entity in spec.geometry.entities:
        if entity.type == "box":
            size = _vec3(entity.size.magnitude)
            origin = _vec3(entity.origin.magnitude) if entity.origin else (0.0, 0.0, 0.0)
            tag = gmsh.model.occ.addBox(origin[0], origin[1], origin[2], size[0], size[1], size[2])
        else:
            radius = _scalar(entity.radius.magnitude)
            height = _scalar(entity.height.magnitude)
            axis = _axis(entity.axis or (0.0, 0.0, 1.0))
            tag = gmsh.model.occ.addCylinder(0.0, 0.0, 0.0, axis[0] * height, axis[1] * height, axis[2] * height, radius)
        entities[entity.name] = GeometryEntityIR(
            name=entity.name,
            type=entity.type,
            dim=3,
            backend_tag=tag,
            bounds=_occ_bounds(3, tag),
        )
    return entities


def _build_tag_map(spec: ProblemSpec, entities: dict[str, GeometryEntityIR]) -> TagMap:
    diagnostics: list[Diagnostic] = []
    bindings: list[TagBinding] = []
    physical_id = 1
    for namespace in _NAMESPACES:
        tags = getattr(spec.tags, namespace)
        for name, tag in sorted(tags.items()):
            selected = _select_entities(tag, entities, ("tags", namespace, name), diagnostics)
            if not selected:
                diagnostics.append(
                    Diagnostic(
                        code="TAG_EMPTY_SELECTION",
                        message=f"Tag '{name}' selected no entities.",
                        path=("tags", namespace, name),
                        source="tags",
                    )
                )
                continue
            bindings.append(
                TagBinding(
                    namespace=namespace,
                    name=name,
                    dim=tag.dim,
                    entity_tags=tuple(sorted(selected)),
                    physical_id=physical_id,
                    physical_name=f"{namespace}/{name}",
                )
            )
            physical_id += 1
    if diagnostics:
        raise MeshError(diagnostics)
    return TagMap(bindings=tuple(bindings))


def _select_entities(
    tag: SemanticTagSpec,
    entities: dict[str, GeometryEntityIR],
    path: tuple[str | int, ...],
    diagnostics: list[Diagnostic],
) -> set[int]:
    selected: set[int] = set()
    for name in tag.entities:
        selected.add(entities[name].backend_tag)
    if tag.selector:
        selector = _parse_selector(tag.selector, path, diagnostics)
        if selector is not None:
            for dim, backend_tag in gmsh.model.getEntities(tag.dim):
                if _selector_matches(selector, _bounds(dim, backend_tag)):
                    selected.add(backend_tag)
    return selected


def _apply_sizes(spec: ProblemSpec, entities: dict[str, GeometryEntityIR], tags: TagMap) -> None:
    global_size = _scalar(spec.mesh.size.global_size.magnitude)
    points = gmsh.model.getEntities(0)
    if points:
        gmsh.model.mesh.setSize(points, global_size)
    bindings = {binding.name: binding for binding in tags.bindings} | {binding.physical_name: binding for binding in tags.bindings}
    diagnostics: list[Diagnostic] = []
    for name, quantity in spec.mesh.size.local.items():
        size = _scalar(quantity.magnitude)
        targets: list[tuple[int, int]] = []
        if name in entities:
            targets = gmsh.model.getBoundary([(entities[name].dim, entities[name].backend_tag)], combined=False, oriented=False, recursive=True)
        elif name in bindings:
            binding = bindings[name]
            targets = gmsh.model.getBoundary([(binding.dim, tag) for tag in binding.entity_tags], combined=False, oriented=False, recursive=True)
        if not targets:
            diagnostics.append(
                Diagnostic(
                    code="MESH_LOCAL_SIZE_TARGET_UNKNOWN",
                    message=f"Local mesh size target '{name}' is not a geometry entity or tag.",
                    path=("mesh", "size", "local", name),
                    source="mesh",
                )
            )
            continue
        gmsh.model.mesh.setSize([target for target in targets if target[0] == 0], size)
    if diagnostics:
        raise MeshError(diagnostics)


def _metadata(spec: ProblemSpec, mesh_path: Path, entities: dict[str, GeometryEntityIR], tags: TagMap) -> MeshMetadata:
    node_tags, _, _ = gmsh.model.mesh.getNodes()
    _, element_tags, _ = gmsh.model.mesh.getElements(spec.mesh.dimension)
    return MeshMetadata(
        backend="gmsh_occ",
        format="msh4",
        path=str(mesh_path),
        dimension=spec.mesh.dimension,
        cell_type=spec.mesh.cell_type,
        order=spec.mesh.order,
        nodes=len(node_tags),
        elements=sum(len(tags) for tags in element_tags),
        entities=tuple(sorted(entities.values(), key=lambda entity: entity.name)),
        tags=tags,
        physical_groups={binding.physical_name: binding.physical_id for binding in tags.bindings},
    )


def _parse_selector(
    selector: str,
    path: tuple[str | int, ...],
    diagnostics: list[Diagnostic],
) -> tuple[str, str, float] | None:
    match = _SELECTOR.match(selector)
    if match is None:
        diagnostics.append(
            Diagnostic(
                code="TAG_SELECTOR_INVALID",
                message=f"Selector '{selector}' is not supported.",
                path=(*path, "selector"),
                source="tags",
            )
        )
        return None
    axis, op, value = match.groups()
    try:
        return axis, op, _selector_value(value.strip())
    except Exception as exc:
        diagnostics.append(
            Diagnostic(
                code="TAG_SELECTOR_INVALID",
                message=f"Selector '{selector}' has an invalid coordinate value.",
                path=(*path, "selector"),
                source="tags",
                backend_error=str(exc),
            )
        )
        return None


def _selector_value(value: str) -> float:
    if _NUMBER.match(value):
        return float(value)
    quantity = canonical_quantity(value, ("tags", "selector"), "[length]")
    return _scalar(quantity.magnitude)


def _selector_matches(selector: tuple[str, str, float], bounds: tuple[float, float, float, float, float, float]) -> bool:
    axis, op, value = selector
    index = {"x": 0, "y": 1, "z": 2}[axis]
    lo = bounds[index]
    hi = bounds[index + 3]
    center = 0.5 * (lo + hi)
    tolerance = max(1.0e-6, 1.0e-8 * max(1.0, abs(value), abs(lo), abs(hi)))
    if op == "==":
        return abs(lo - value) <= tolerance and abs(hi - value) <= tolerance
    if op == "<":
        return center < value - tolerance
    if op == ">":
        return center > value + tolerance
    if op == "<=":
        return center <= value + tolerance
    return center >= value - tolerance


def _cell_type_diagnostics(spec: ProblemSpec) -> list[Diagnostic]:
    supported = {
        1: {"interval"},
        2: {"triangle"},
        3: {"tetrahedron"},
    }
    if spec.mesh.cell_type not in supported[spec.mesh.dimension]:
        return [
            Diagnostic(
                code="MESH_CELL_TYPE_UNSUPPORTED",
                message=f"Gmsh mesh generation for {spec.mesh.cell_type} cells is not implemented.",
                path=("mesh", "cell_type"),
                source="mesh",
            )
        ]
    return []


def _vec3(value: Any) -> tuple[float, float, float]:
    if not isinstance(value, tuple) or len(value) != 3:
        raise ValueError("Expected 3-vector magnitude")
    return (float(value[0]), float(value[1]), float(value[2]))


def _scalar(value: Any) -> float:
    if isinstance(value, tuple):
        raise ValueError("Expected scalar magnitude")
    return float(value)


def _axis(axis: tuple[float, float, float]) -> tuple[float, float, float]:
    length = math.sqrt(axis[0] * axis[0] + axis[1] * axis[1] + axis[2] * axis[2])
    if length == 0.0:
        raise ValueError("Cylinder axis must be nonzero")
    return (axis[0] / length, axis[1] / length, axis[2] / length)


def _bounds(dim: int, tag: int) -> tuple[float, float, float, float, float, float]:
    return tuple(float(value) for value in gmsh.model.getBoundingBox(dim, tag))


def _occ_bounds(dim: int, tag: int) -> tuple[float, float, float, float, float, float]:
    return tuple(float(value) for value in gmsh.model.occ.getBoundingBox(dim, tag))
