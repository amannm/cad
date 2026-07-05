from __future__ import annotations

from copy import deepcopy
from typing import Any, Callable, Iterable

from cadmultiphysics.core import build_domain_ir, build_problem_spec, build_run_plan
from cadmultiphysics.run import CommandResult, solve_spec
from cadmultiphysics.schema import DomainIR, ProblemSpec, RunPlan


class GeometryBuilder:
    def __init__(self) -> None:
        self.entities: list[dict[str, Any]] = []

    def box(
        self,
        name: str,
        *,
        x: Any | None = None,
        y: Any | None = None,
        z: Any | None = None,
        size: Iterable[Any] | None = None,
        origin: Iterable[Any] | None = None,
    ) -> str:
        values = tuple(size) if size is not None else (x, y, z)
        if len(values) != 3 or any(value is None for value in values):
            raise ValueError("box requires size or x, y, z")
        entity: dict[str, Any] = {"type": "box", "name": name, "size": values}
        if origin is not None:
            entity["origin"] = tuple(origin)
        self.entities.append(entity)
        return name

    def cylinder(
        self,
        name: str,
        *,
        radius: Any,
        height: Any,
        axis: Iterable[float] = (0.0, 0.0, 1.0),
    ) -> str:
        self.entities.append(
            {
                "type": "cylinder",
                "name": name,
                "radius": radius,
                "height": height,
                "axis": tuple(axis),
            }
        )
        return name

    def boolean_union(self, name: str, *, entities: Iterable[str]) -> str:
        self.entities.append({"type": "boolean_union", "name": name, "entities": tuple(entities)})
        return name

    def boolean_cut(self, name: str, *, base: str, tools: Iterable[str]) -> str:
        self.entities.append({"type": "boolean_cut", "name": name, "base": base, "tools": tuple(tools)})
        return name


class Problem:
    def __init__(self, name: str) -> None:
        self._data: dict[str, Any] = {
            "name": name,
            "version": "0.1.0",
            "units": {"system": "SI"},
            "tags": {"materials": {}, "boundaries": {}, "interfaces": {}, "curves": {}, "points": {}},
            "fields": [],
            "materials": {},
            "boundary_conditions": [],
            "loads": [],
            "initial_conditions": [],
            "solver": {},
            "output": {},
            "metadata": {},
        }

    def mode(self, mode: str) -> "Problem":
        self._data["mode"] = mode
        return self

    def geometry(self, build: Callable[[GeometryBuilder], Any]) -> "Problem":
        builder = GeometryBuilder()
        build(builder)
        self._data["geometry"] = {"backend": "gmsh_occ", "entities": builder.entities}
        return self

    def tag(
        self,
        namespace: str,
        name: str,
        *,
        dim: int,
        entities: Iterable[str] = (),
        selector: str | None = None,
    ) -> "Problem":
        self._data["tags"][namespace][name] = {"dim": dim, "entities": list(entities), "selector": selector}
        return self

    def boundary(self, name: str, *, selector: str | None = None, entities: Iterable[str] = (), dim: int = 2) -> "Problem":
        return self.tag("boundaries", name, dim=dim, entities=entities, selector=selector)

    def interface(self, name: str, *, selector: str | None = None, entities: Iterable[str] = (), dim: int = 2) -> "Problem":
        return self.tag("interfaces", name, dim=dim, entities=entities, selector=selector)

    def curve(self, name: str, *, selector: str | None = None, entities: Iterable[str] = (), dim: int = 1) -> "Problem":
        return self.tag("curves", name, dim=dim, entities=entities, selector=selector)

    def point(self, name: str, *, selector: str | None = None, entities: Iterable[str] = (), dim: int = 0) -> "Problem":
        return self.tag("points", name, dim=dim, entities=entities, selector=selector)

    def material(self, name: str, *, domain: str | Iterable[str], model: str, params: dict[str, Any] | None = None) -> "Problem":
        entities = (domain,) if isinstance(domain, str) else tuple(domain)
        self._data["tags"]["materials"][name] = {"dim": 3, "entities": list(entities)}
        self._data["materials"][name] = {"model": model, "parameters": params or {}}
        return self

    def field(
        self,
        name: str,
        *,
        unit: Any,
        family: str = "Lagrange",
        order: int = 1,
        kind: str | None = None,
        components: int | None = None,
    ) -> "Problem":
        field_kind = kind or ("vector" if components and components > 1 else "scalar")
        payload: dict[str, Any] = {
            "name": name,
            "kind": field_kind,
            "unit": _unit_name(unit),
            "element": {"family": family, "order": order},
        }
        if components is not None:
            payload["components"] = components
        self._data["fields"].append(payload)
        return self

    def bc(
        self,
        name: str,
        *,
        on: str,
        field: str,
        type: str = "dirichlet",
        value: Any | None = None,
        **parameters: Any,
    ) -> "Problem":
        self._data["boundary_conditions"].append(
            {"name": name, "on": on, "field": field, "type": type, "value": value, "parameters": parameters}
        )
        return self

    def load(
        self,
        name: str,
        *,
        on: str,
        field: str,
        type: str = "source",
        value: Any | None = None,
        flux: Any | None = None,
        **parameters: Any,
    ) -> "Problem":
        self._data["loads"].append(
            {
                "name": name,
                "on": on,
                "field": field,
                "type": "flux" if flux is not None else type,
                "value": flux if flux is not None else value,
                "parameters": parameters,
            }
        )
        return self

    def initial_condition(self, *, field: str, value: Any) -> "Problem":
        self._data["initial_conditions"].append({"field": field, "value": value})
        return self

    def time(self, *, start: Any, stop: Any, step: Any, scheme: str = "backward_euler") -> "Problem":
        self._data["time"] = {"start": start, "stop": stop, "step": step, "scheme": scheme}
        return self

    def mesh(
        self,
        *,
        cell_type: str = "tetrahedron",
        order: int = 1,
        global_size: Any,
        dimension: int | None = None,
        curvature: bool = False,
        partitions: int = 1,
        local: dict[str, Any] | None = None,
        quality: dict[str, Any] | None = None,
    ) -> "Problem":
        payload: dict[str, Any] = {
            "cell_type": cell_type,
            "order": order,
            "size": {"global": global_size, "local": local or {}},
            "curvature": curvature,
            "partitions": partitions,
            "quality": quality or {},
        }
        if dimension is not None:
            payload["dimension"] = dimension
        self._data["mesh"] = payload
        return self

    def solver(
        self,
        *,
        prefix: str = "",
        linear: dict[str, Any] | None = None,
        nonlinear: dict[str, Any] | None = None,
        fieldsplits: dict[str, dict[str, Any]] | None = None,
        allow_backend_options: bool = False,
    ) -> "Problem":
        self._data["solver"] = {
            "prefix": prefix,
            "linear": linear or {},
            "nonlinear": nonlinear or {},
            "fieldsplits": fieldsplits or {},
            "allow_backend_options": allow_backend_options,
        }
        return self

    def output(
        self,
        *,
        fields: Iterable[str] = (),
        derived_fields: Iterable[str] = (),
        format: str = "vtx",
        cadence: str = "end",
        restart_cadence: str | None = None,
        report_formats: Iterable[str] = ("json",),
        writer_options: dict[str, Any] | None = None,
    ) -> "Problem":
        self._data["output"] = {
            "format": format,
            "fields": list(fields),
            "derived_fields": list(derived_fields),
            "cadence": cadence,
            "restart_cadence": restart_cadence,
            "report_formats": list(report_formats),
            "writer_options": writer_options or {},
        }
        return self

    def metadata(self, **metadata: Any) -> "Problem":
        self._data["metadata"] = dict(sorted(metadata.items()))
        return self

    def data(self) -> dict[str, Any]:
        return deepcopy(self._data)

    def spec(self) -> ProblemSpec:
        return build_problem_spec(self.data())

    def domain(self) -> DomainIR:
        return build_domain_ir(self.spec())

    def run_plan(self) -> RunPlan:
        return build_run_plan(self.spec())

    def run(self, out: str) -> CommandResult:
        return solve_spec(self.spec(), out)


def _unit_name(unit: Any) -> str:
    if hasattr(unit, "units"):
        return str(unit.units)
    return str(unit)
