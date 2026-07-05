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
_AND = re.compile(r"\s+and\s+", re.IGNORECASE)
_NUMBER = re.compile(r"^[+-]?(?:\d+(?:\.\d*)?|\.\d+)(?:[eE][+-]?\d+)?$")
_NAMESPACES: tuple[TagNamespace, ...] = ("materials", "boundaries", "interfaces", "curves", "points")


@dataclass(frozen=True)
class MeshBuild:
    metadata: MeshMetadata
    artifacts: dict[str, Path]


@dataclass(frozen=True)
class QualityThreshold:
    measure: str
    minimum: float | None
    maximum: float | None
    path: tuple[str | int, ...]


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
        quality_report = _quality_report(spec)
        mesh_path = mesh_dir / "model.msh"
        gmsh.write(str(mesh_path))
        metadata = _metadata(spec, mesh_path, entities, tags, quality_report)
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
            entities[entity.name] = GeometryEntityIR(
                name=entity.name,
                type=entity.type,
                dim=3,
                backend_tag=tag,
                bounds=_occ_bounds(3, tag),
            )
        elif entity.type == "cylinder":
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
        elif entity.type == "boolean_union":
            out = _boolean_result(
                entity.name,
                entity.type,
                gmsh.model.occ.fuse([_entity_dim_tag(entities, entity.entities[0])], _entity_dim_tags(entities, entity.entities[1:])),
            )
            for name in entity.entities:
                entities.pop(name, None)
            entities[entity.name] = out
        else:
            out = _boolean_result(
                entity.name,
                entity.type,
                gmsh.model.occ.cut([_entity_dim_tag(entities, entity.base)], _entity_dim_tags(entities, entity.tools)),
            )
            entities.pop(entity.base, None)
            for name in entity.tools:
                entities.pop(name, None)
            entities[entity.name] = out
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
        entity = entities.get(name)
        if entity is None:
            diagnostics.append(
                Diagnostic(
                    code="TAG_ENTITY_INACTIVE",
                    message=f"Tag references geometry entity '{name}' that is not active after boolean operations.",
                    path=(*path, "entities"),
                    source="tags",
                )
            )
            continue
        selected.add(entity.backend_tag)
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
            bound = [(binding.dim, tag) for tag in binding.entity_tags]
            targets = bound if binding.dim == 0 else gmsh.model.getBoundary(bound, combined=False, oriented=False, recursive=True)
        points = [target for target in targets if target[0] == 0]
        if not points:
            diagnostics.append(
                Diagnostic(
                    code="MESH_LOCAL_SIZE_TARGET_UNKNOWN",
                    message=f"Local mesh size target '{name}' does not resolve to mesh points.",
                    path=("mesh", "size", "local", name),
                    source="mesh",
                )
            )
            continue
        gmsh.model.mesh.setSize(points, size)
    if diagnostics:
        raise MeshError(diagnostics)


def _metadata(
    spec: ProblemSpec,
    mesh_path: Path,
    entities: dict[str, GeometryEntityIR],
    tags: TagMap,
    quality_report: dict[str, Any],
) -> MeshMetadata:
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
        physical_names={binding.physical_id: binding.physical_name for binding in tags.bindings},
        partition={"requested": spec.mesh.partitions, "actual": 1},
        quality_report=quality_report,
    )


def _quality_report(spec: ProblemSpec) -> dict[str, Any]:
    element_tags = _element_tags(spec.mesh.dimension)
    if not element_tags:
        raise MeshError(
            [
                Diagnostic(
                    code="MESH_EMPTY",
                    message="Generated mesh contains no top-dimensional elements.",
                    path=("mesh",),
                    source="mesh",
                )
            ]
        )
    thresholds = _quality_thresholds(spec.mesh.quality)
    measures = _quality_measures(spec.mesh.dimension, thresholds)
    stats: dict[str, dict[str, float | int]] = {}
    diagnostics: list[Diagnostic] = []
    for measure in measures:
        try:
            values = [float(value) for value in gmsh.model.mesh.getElementQualities(element_tags, measure)]
        except Exception as exc:
            diagnostics.append(
                Diagnostic(
                    code="MESH_QUALITY_MEASURE_FAILED",
                    message=f"Could not compute mesh quality measure '{measure}'.",
                    path=("mesh", "quality", measure),
                    source="mesh",
                    backend_error=str(exc),
                )
            )
            continue
        stats[measure] = {
            "count": len(values),
            "min": min(values),
            "max": max(values),
            "mean": sum(values) / len(values),
        }
    for threshold in thresholds:
        observed = stats.get(threshold.measure)
        if observed is None:
            continue
        if threshold.minimum is not None and float(observed["min"]) < threshold.minimum:
            diagnostics.append(
                Diagnostic(
                    code="MESH_QUALITY_THRESHOLD_FAILED",
                    message=f"Mesh quality '{threshold.measure}' minimum is below threshold.",
                    path=threshold.path,
                    source="mesh",
                    payload={"observed": observed["min"], "threshold": threshold.minimum},
                )
            )
        if threshold.maximum is not None and float(observed["max"]) > threshold.maximum:
            diagnostics.append(
                Diagnostic(
                    code="MESH_QUALITY_THRESHOLD_FAILED",
                    message=f"Mesh quality '{threshold.measure}' maximum is above threshold.",
                    path=threshold.path,
                    source="mesh",
                    payload={"observed": observed["max"], "threshold": threshold.maximum},
                )
            )
    if diagnostics:
        raise MeshError(diagnostics)
    return {
        "elements": len(element_tags),
        "measures": stats,
        "thresholds": [
            {"measure": threshold.measure, "min": threshold.minimum, "max": threshold.maximum}
            for threshold in thresholds
        ],
    }


def _element_tags(dimension: int) -> list[int]:
    _, element_tags, _ = gmsh.model.mesh.getElements(dimension)
    return [int(tag) for block in element_tags for tag in block]


def _quality_measures(dimension: int, thresholds: tuple[QualityThreshold, ...]) -> tuple[str, ...]:
    defaults = {
        1: ("minEdge", "maxEdge"),
        2: ("minSICN", "gamma", "minEdge", "maxEdge", "volume"),
        3: ("minSICN", "gamma", "minEdge", "maxEdge", "volume"),
    }
    return tuple(dict.fromkeys((*defaults[dimension], *(threshold.measure for threshold in thresholds))))


def _quality_thresholds(config: dict[str, Any]) -> tuple[QualityThreshold, ...]:
    thresholds: list[QualityThreshold] = []
    for measure, value in sorted(config.items()):
        path = ("mesh", "quality", measure)
        if isinstance(value, dict) and ("min" in value or "max" in value or "measure" in value):
            thresholds.append(
                QualityThreshold(
                    measure=str(value.get("measure", measure)),
                    minimum=_threshold_value(value.get("min"), (*path, "min")) if value.get("min") is not None else None,
                    maximum=_threshold_value(value.get("max"), (*path, "max")) if value.get("max") is not None else None,
                    path=path,
                )
            )
        else:
            thresholds.append(QualityThreshold(measure=measure, minimum=_threshold_value(value, path), maximum=None, path=path))
    return tuple(thresholds)


def _threshold_value(value: Any, path: tuple[str | int, ...]) -> float:
    if isinstance(value, int | float):
        return float(value)
    if isinstance(value, dict) and "magnitude" in value:
        magnitude = value["magnitude"]
        if isinstance(magnitude, int | float):
            return float(magnitude)
    raise MeshError(
        [
            Diagnostic(
                code="MESH_QUALITY_THRESHOLD_INVALID",
                message="Mesh quality threshold must be scalar.",
                path=path,
                source="mesh",
            )
        ]
    )


def _parse_selector(
    selector: str,
    path: tuple[str | int, ...],
    diagnostics: list[Diagnostic],
) -> tuple[tuple[str, str, float], ...] | None:
    clauses: list[tuple[str, str, float]] = []
    for index, clause in enumerate(_AND.split(selector)):
        match = _SELECTOR.match(clause)
        if match is None:
            diagnostics.append(
                Diagnostic(
                    code="TAG_SELECTOR_INVALID",
                    message=f"Selector '{selector}' contains unsupported clause '{clause}'.",
                    path=(*path, "selector", index),
                    source="tags",
                )
            )
            return None
        axis, op, value = match.groups()
        try:
            clauses.append((axis, op, _selector_value(value.strip(), (*path, "selector", index))))
        except Exception as exc:
            diagnostics.append(
                Diagnostic(
                    code="TAG_SELECTOR_INVALID",
                    message=f"Selector '{selector}' has an invalid coordinate value.",
                    path=(*path, "selector", index),
                    source="tags",
                    backend_error=str(exc),
                )
            )
            return None
    return tuple(clauses)


def _selector_value(value: str, path: tuple[str | int, ...]) -> float:
    if _NUMBER.match(value):
        return float(value)
    quantity = canonical_quantity(value, path, "[length]")
    return _scalar(quantity.magnitude)


def _selector_matches(selector: tuple[tuple[str, str, float], ...], bounds: tuple[float, float, float, float, float, float]) -> bool:
    return all(_selector_clause_matches(clause, bounds) for clause in selector)


def _selector_clause_matches(clause: tuple[str, str, float], bounds: tuple[float, float, float, float, float, float]) -> bool:
    axis, op, value = clause
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
    diagnostics: list[Diagnostic] = []
    supported = {
        1: {"interval"},
        2: {"triangle"},
        3: {"tetrahedron"},
    }
    if spec.mesh.cell_type not in supported[spec.mesh.dimension]:
        diagnostics.append(
            Diagnostic(
                code="MESH_CELL_TYPE_UNSUPPORTED",
                message=f"Gmsh mesh generation for {spec.mesh.cell_type} cells is not implemented.",
                path=("mesh", "cell_type"),
                source="mesh",
            )
        )
    if spec.mesh.partitions != 1:
        diagnostics.append(
            Diagnostic(
                code="MESH_PARTITIONING_UNSUPPORTED",
                message="Gmsh mesh partitioning is not implemented in this mesh generator.",
                path=("mesh", "partitions"),
                source="mesh",
            )
        )
    return diagnostics


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


def _entity_dim_tag(entities: dict[str, GeometryEntityIR], name: str) -> tuple[int, int]:
    entity = entities.get(name)
    if entity is None:
        raise MeshError(
            [
                Diagnostic(
                    code="GEOMETRY_REFERENCE_INACTIVE",
                    message=f"Geometry operation references inactive entity '{name}'.",
                    path=("geometry", "entities"),
                    source="geometry",
                )
            ]
        )
    return (entity.dim, entity.backend_tag)


def _entity_dim_tags(entities: dict[str, GeometryEntityIR], names: tuple[str, ...]) -> list[tuple[int, int]]:
    return [_entity_dim_tag(entities, name) for name in names]


def _boolean_result(name: str, entity_type: str, result: tuple[Any, Any]) -> GeometryEntityIR:
    out_dim_tags, _ = result
    volumes = [(int(dim), int(tag)) for dim, tag in out_dim_tags if int(dim) == 3]
    if len(volumes) != 1:
        raise MeshError(
            [
                Diagnostic(
                    code="GEOMETRY_BOOLEAN_RESULT_INVALID",
                    message=f"Boolean entity '{name}' must produce one volume.",
                    path=("geometry", "entities", name),
                    source="geometry",
                    payload={"volumes": volumes},
                )
            ]
        )
    dim, tag = volumes[0]
    return GeometryEntityIR(name=name, type=entity_type, dim=dim, backend_tag=tag, bounds=_occ_bounds(dim, tag))
