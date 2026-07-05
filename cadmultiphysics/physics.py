from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any

from cadmultiphysics.diagnostics import Diagnostic
from cadmultiphysics.schema import FieldSpec, ProblemSpec
from cadmultiphysics.units import dimension_of

DIMENSIONLESS = "dimensionless"
LENGTH = "[length]"
TEMPERATURE = "[temperature]"
PRESSURE = dimension_of("Pa")
THERMAL_CONDUCTIVITY = dimension_of("W/(m*K)")
HEAT_FLUX = dimension_of("W/m^2")
DENSITY = dimension_of("kg/m^3")
HEAT_CAPACITY = dimension_of("J/(kg*K)")
INV_TEMPERATURE = dimension_of("1/K")


@dataclass(frozen=True)
class ParameterRule:
    dimension: str
    required: bool = True
    positive: bool = False
    lower_open: float | None = None
    upper_open: float | None = None


@dataclass(frozen=True)
class MaterialContract:
    parameters: dict[str, ParameterRule]
    required_fields: tuple[str, ...]
    modes: tuple[str, ...]


MATERIAL_CONTRACTS: dict[str, MaterialContract] = {
    "isotropic_heat": MaterialContract(
        parameters={
            "k": ParameterRule(THERMAL_CONDUCTIVITY, positive=True),
            "rho": ParameterRule(DENSITY, required=False, positive=True),
            "cp": ParameterRule(HEAT_CAPACITY, required=False, positive=True),
        },
        required_fields=("temperature",),
        modes=("linear_steady", "linear_transient"),
    ),
    "thermoelastic_small_strain": MaterialContract(
        parameters={
            "E": ParameterRule(PRESSURE, positive=True),
            "nu": ParameterRule(DIMENSIONLESS, lower_open=-1.0, upper_open=0.5),
            "k": ParameterRule(THERMAL_CONDUCTIVITY, positive=True),
            "alpha": ParameterRule(INV_TEMPERATURE, required=False),
            "T0": ParameterRule(TEMPERATURE, required=False),
            "rho": ParameterRule(DENSITY, required=False, positive=True),
            "cp": ParameterRule(HEAT_CAPACITY, required=False, positive=True),
        },
        required_fields=("displacement", "temperature"),
        modes=("linear_steady", "linear_transient", "nonlinear_steady", "nonlinear_transient"),
    ),
    "linear_elastic_small_strain": MaterialContract(
        parameters={
            "E": ParameterRule(PRESSURE, positive=True),
            "nu": ParameterRule(DIMENSIONLESS, lower_open=-1.0, upper_open=0.5),
            "rho": ParameterRule(DENSITY, required=False, positive=True),
        },
        required_fields=("displacement",),
        modes=("linear_steady", "linear_transient"),
    ),
}


def physics_diagnostics(spec: ProblemSpec) -> list[Diagnostic]:
    diagnostics: list[Diagnostic] = []
    field_roles = _field_roles(spec.fields)
    for name, material in spec.materials.items():
        contract = MATERIAL_CONTRACTS.get(material.model)
        if contract is None:
            diagnostics.append(
                Diagnostic(
                    code="PHYSICS_MATERIAL_MODEL_UNKNOWN",
                    message=f"Material '{name}' uses unknown model '{material.model}'.",
                    path=("materials", name, "model"),
                    source="physics",
                )
            )
            continue
        if spec.mode not in contract.modes:
            diagnostics.append(
                Diagnostic(
                    code="PHYSICS_MODE_UNSUPPORTED",
                    message=f"Material model '{material.model}' does not support mode '{spec.mode}'.",
                    path=("materials", name, "model"),
                    source="physics",
                )
            )
        for role in contract.required_fields:
            if role not in field_roles:
                diagnostics.append(
                    Diagnostic(
                        code="PHYSICS_FIELD_REQUIRED",
                        message=f"Material model '{material.model}' requires a {role} field.",
                        path=("fields",),
                        source="physics",
                    )
                )
        diagnostics.extend(_parameter_diagnostics(name, material.model, material.parameters, contract.parameters))
        diagnostics.extend(_transient_material_diagnostics(spec.mode, name, material.model, material.parameters))
        diagnostics.extend(_reference_temperature_diagnostics(spec.mode, name, material.model, material.parameters))
    diagnostics.extend(_bc_diagnostics(spec))
    diagnostics.extend(_load_diagnostics(spec, field_roles))
    return diagnostics


def _field_roles(fields: tuple[FieldSpec, ...]) -> dict[str, tuple[str, ...]]:
    roles: dict[str, list[str]] = {}
    for field in fields:
        if field.kind == "scalar" and field.unit.dimension == TEMPERATURE:
            roles.setdefault("temperature", []).append(field.name)
        if field.kind == "vector" and field.unit.dimension == LENGTH:
            roles.setdefault("displacement", []).append(field.name)
    return {role: tuple(names) for role, names in roles.items()}


def _parameter_diagnostics(
    material_name: str,
    model: str,
    parameters: dict[str, Any],
    rules: dict[str, ParameterRule],
) -> list[Diagnostic]:
    diagnostics: list[Diagnostic] = []
    for key, rule in sorted(rules.items()):
        if rule.required and key not in parameters:
            diagnostics.append(
                Diagnostic(
                    code="PHYSICS_PARAMETER_REQUIRED",
                    message=f"Material model '{model}' requires parameter '{key}'.",
                    path=("materials", material_name, "parameters", key),
                    source="physics",
                )
            )
    for key, value in sorted(parameters.items()):
        rule = rules.get(key)
        if rule is None:
            diagnostics.append(
                Diagnostic(
                    code="PHYSICS_PARAMETER_UNKNOWN",
                    message=f"Material model '{model}' does not declare parameter '{key}'.",
                    path=("materials", material_name, "parameters", key),
                    source="physics",
                )
            )
            continue
        diagnostics.extend(_quantity_rule_diagnostics(value, rule, ("materials", material_name, "parameters", key), f"Parameter '{key}'"))
    return diagnostics


def _transient_material_diagnostics(mode: str, material_name: str, model: str, parameters: dict[str, Any]) -> list[Diagnostic]:
    if mode not in {"linear_transient", "nonlinear_transient"} or model not in {"isotropic_heat", "thermoelastic_small_strain"}:
        return []
    return [
        Diagnostic(
            code="PHYSICS_PARAMETER_REQUIRED",
            message=f"Material model '{model}' requires parameter '{key}' for mode '{mode}'.",
            path=("materials", material_name, "parameters", key),
            source="physics",
        )
        for key in ("rho", "cp")
        if key not in parameters
    ]


def _reference_temperature_diagnostics(mode: str, material_name: str, model: str, parameters: dict[str, Any]) -> list[Diagnostic]:
    if not mode.endswith("_steady") or model != "thermoelastic_small_strain" or "alpha" not in parameters or "T0" in parameters:
        return []
    return [
        Diagnostic(
            code="PHYSICS_PARAMETER_REQUIRED",
            message=f"Material model '{model}' requires parameter 'T0' when steady thermal expansion is enabled.",
            path=("materials", material_name, "parameters", "T0"),
            source="physics",
        )
    ]


def _bc_diagnostics(spec: ProblemSpec) -> list[Diagnostic]:
    diagnostics: list[Diagnostic] = []
    known = {"dirichlet", "neumann", "robin"}
    for index, bc in enumerate(spec.bcs):
        path = ("boundary_conditions", index)
        if bc.type not in known:
            diagnostics.append(
                Diagnostic(
                    code="PHYSICS_BC_TYPE_UNKNOWN",
                    message=f"Boundary condition '{bc.name}' has unknown type '{bc.type}'.",
                    path=(*path, "type"),
                    source="physics",
                )
            )
        if bc.type == "dirichlet" and bc.value is None:
            diagnostics.append(
                Diagnostic(
                    code="PHYSICS_BC_VALUE_REQUIRED",
                    message=f"Dirichlet boundary condition '{bc.name}' requires a value.",
                    path=(*path, "value"),
                    source="physics",
                )
            )
    return diagnostics


def _load_diagnostics(spec: ProblemSpec, field_roles: dict[str, tuple[str, ...]]) -> list[Diagnostic]:
    diagnostics: list[Diagnostic] = []
    temperature_fields = set(field_roles.get("temperature", ()))
    for index, load in enumerate(spec.loads):
        path = ("loads", index)
        if load.type not in {"source", "flux", "body_force", "traction"}:
            diagnostics.append(
                Diagnostic(
                    code="PHYSICS_LOAD_TYPE_UNKNOWN",
                    message=f"Load '{load.name}' has unknown type '{load.type}'.",
                    path=(*path, "type"),
                    source="physics",
                )
            )
        if load.value is None:
            diagnostics.append(
                Diagnostic(
                    code="PHYSICS_LOAD_VALUE_REQUIRED",
                    message=f"Load '{load.name}' requires a value.",
                    path=(*path, "value"),
                    source="physics",
                )
            )
            continue
        if load.type == "flux" and load.field in temperature_fields:
            diagnostics.extend(
                _quantity_rule_diagnostics(
                    load.value,
                    ParameterRule(HEAT_FLUX),
                    (*path, "value"),
                    f"Heat flux load '{load.name}'",
                )
            )
    return diagnostics


def _quantity_rule_diagnostics(value: Any, rule: ParameterRule, path: tuple[str | int, ...], subject: str) -> list[Diagnostic]:
    if not isinstance(value, dict) or "dimension" not in value or "magnitude" not in value:
        return [
            Diagnostic(
                code="PHYSICS_QUANTITY_REQUIRED",
                message=f"{subject} must be a canonical quantity.",
                path=path,
                source="physics",
            )
        ]
    diagnostics: list[Diagnostic] = []
    dimension = str(value["dimension"])
    magnitude = value["magnitude"]
    if dimension != rule.dimension:
        diagnostics.append(
            Diagnostic(
                code="PHYSICS_PARAMETER_DIMENSION_MISMATCH",
                message=f"{subject} must have dimension {rule.dimension}, got {dimension}.",
                path=path,
                source="physics",
            )
        )
    if not isinstance(magnitude, int | float) or isinstance(magnitude, bool) or not math.isfinite(float(magnitude)):
        diagnostics.append(
            Diagnostic(
                code="PHYSICS_PARAMETER_SCALAR_REQUIRED",
                message=f"{subject} must be a finite scalar.",
                path=path,
                source="physics",
            )
        )
        return diagnostics
    scalar = float(magnitude)
    if rule.positive and scalar <= 0.0:
        diagnostics.append(
            Diagnostic(
                code="PHYSICS_PARAMETER_POSITIVE_REQUIRED",
                message=f"{subject} must be positive.",
                path=path,
                source="physics",
            )
        )
    if rule.lower_open is not None and scalar <= rule.lower_open:
        diagnostics.append(
            Diagnostic(
                code="PHYSICS_PARAMETER_RANGE_INVALID",
                message=f"{subject} must be greater than {rule.lower_open}.",
                path=path,
                source="physics",
            )
        )
    if rule.upper_open is not None and scalar >= rule.upper_open:
        diagnostics.append(
            Diagnostic(
                code="PHYSICS_PARAMETER_RANGE_INVALID",
                message=f"{subject} must be less than {rule.upper_open}.",
                path=path,
                source="physics",
            )
        )
    return diagnostics
