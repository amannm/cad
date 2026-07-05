from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any

from pydantic import BaseModel, ConfigDict
from pint import DimensionalityError, UndefinedUnitError, UnitRegistry

from cadmultiphysics.diagnostics import Diagnostic

ureg = UnitRegistry(autoconvert_offset_to_baseunit=True)


class QuantitySpec(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    magnitude: float | tuple[float, ...]
    unit: str
    dimension: str


class UnitSpec(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    unit: str
    dimension: str


class UnitDiagnostics(Exception):
    def __init__(self, diagnostics: list[Diagnostic]):
        self.diagnostics = diagnostics
        super().__init__(diagnostics[0].message if diagnostics else "Unit diagnostics")


def unit_spec(value: Any, path: Sequence[str | int] = ()) -> UnitSpec:
    unit = value if isinstance(value, str) else str(value)
    try:
        quantity = (1 * ureg.Unit(unit)).to_base_units()
    except (UndefinedUnitError, ValueError) as exc:
        raise UnitDiagnostics(
            [
                Diagnostic(
                    code="UNIT_UNKNOWN",
                    message=f"Unknown unit '{unit}'.",
                    path=tuple(path),
                    source="units",
                    backend_error=str(exc),
                )
            ]
        ) from exc
    return UnitSpec(unit=_unit(quantity), dimension=str(quantity.dimensionality))


def canonical_quantity(
    value: Any,
    path: Sequence[str | int] = (),
    expected_dimension: str | None = None,
) -> QuantitySpec:
    try:
        if _is_pint_quantity(value):
            return _pint_quantity(value, path, expected_dimension)
        if _is_vector_quantity(value):
            return _vector_quantity(value, path, expected_dimension)
        quantity = _scalar_quantity(value).to_base_units()
    except (DimensionalityError, UndefinedUnitError, ValueError, TypeError) as exc:
        raise UnitDiagnostics(
            [
                Diagnostic(
                    code="UNIT_PARSE_FAILED",
                    message=f"Could not parse quantity at '{_path(path)}'.",
                    path=tuple(path),
                    source="units",
                    backend_error=str(exc),
                )
            ]
        ) from exc
    dimension = str(quantity.dimensionality)
    if expected_dimension is not None and dimension != expected_dimension:
        raise UnitDiagnostics(
            [
                Diagnostic(
                    code="UNIT_DIMENSION_MISMATCH",
                    message=f"Expected dimension {expected_dimension}, got {dimension}.",
                    path=tuple(path),
                    source="units",
                )
            ]
        )
    return QuantitySpec(magnitude=float(quantity.magnitude), unit=_unit(quantity), dimension=dimension)


def canonical_value(value: Any, path: Sequence[str | int] = ()) -> Any:
    if isinstance(value, Mapping):
        return {str(key): canonical_value(inner, (*path, str(key))) for key, inner in sorted(value.items())}
    if _is_pint_quantity(value):
        return canonical_quantity(value, path).model_dump(mode="json")
    if _is_vector_quantity(value):
        return canonical_quantity(value, path).model_dump(mode="json")
    if isinstance(value, Sequence) and not isinstance(value, str):
        return [canonical_value(inner, (*path, index)) for index, inner in enumerate(value)]
    if isinstance(value, bool):
        return value
    if isinstance(value, str):
        return canonical_quantity(value, path).model_dump(mode="json")
    if isinstance(value, int | float):
        return canonical_quantity(value, path).model_dump(mode="json")
    return value


def require_dimension(unit: str, expected: str, path: Sequence[str | int] = ()) -> UnitSpec:
    spec = unit_spec(unit, path)
    if spec.dimension != expected:
        raise UnitDiagnostics(
            [
                Diagnostic(
                    code="UNIT_DIMENSION_MISMATCH",
                    message=f"Expected dimension {expected}, got {spec.dimension}.",
                    path=tuple(path),
                    source="units",
                )
            ]
        )
    return spec


def dimension_of(unit: str) -> str:
    return unit_spec(unit).dimension


def _scalar_quantity(value: Any):
    if _is_pint_quantity(value):
        return value
    if isinstance(value, str):
        return ureg.Quantity(value)
    if isinstance(value, bool):
        raise TypeError("Expected scalar quantity, got bool")
    if isinstance(value, int | float):
        return ureg.Quantity(value)
    raise TypeError(f"Expected scalar quantity, got {type(value).__name__}")


def _vector_quantity(values: Sequence[Any], path: Sequence[str | int], expected_dimension: str | None) -> QuantitySpec:
    if not values:
        raise ValueError("Empty quantity vector")
    quantities = [_scalar_quantity(value).to_base_units() for value in values]
    dimensions = {str(quantity.dimensionality) for quantity in quantities}
    if len(dimensions) != 1:
        raise DimensionalityError(tuple(sorted(dimensions)), "single vector dimension")
    dimension = dimensions.pop()
    if expected_dimension is not None and dimension != expected_dimension:
        raise UnitDiagnostics(
            [
                Diagnostic(
                    code="UNIT_DIMENSION_MISMATCH",
                    message=f"Expected dimension {expected_dimension}, got {dimension}.",
                    path=tuple(path),
                    source="units",
                )
            ]
        )
    return QuantitySpec(
        magnitude=tuple(float(quantity.magnitude) for quantity in quantities),
        unit=_unit(quantities[0]),
        dimension=dimension,
    )


def _pint_quantity(value: Any, path: Sequence[str | int], expected_dimension: str | None) -> QuantitySpec:
    quantity = value.to_base_units()
    dimension = str(quantity.dimensionality)
    if expected_dimension is not None and dimension != expected_dimension:
        raise UnitDiagnostics(
            [
                Diagnostic(
                    code="UNIT_DIMENSION_MISMATCH",
                    message=f"Expected dimension {expected_dimension}, got {dimension}.",
                    path=tuple(path),
                    source="units",
                )
            ]
        )
    return QuantitySpec(magnitude=_magnitude(quantity.magnitude), unit=_unit(quantity), dimension=dimension)


def _is_vector_quantity(value: Any) -> bool:
    return isinstance(value, Sequence) and not isinstance(value, str) and all(
        isinstance(item, str) or (isinstance(item, int | float) and not isinstance(item, bool)) or _is_pint_quantity(item) for item in value
    )


def _is_pint_quantity(value: Any) -> bool:
    return isinstance(value, ureg.Quantity)


def _magnitude(value: Any) -> float | tuple[float, ...]:
    if hasattr(value, "tolist"):
        value = value.tolist()
    if isinstance(value, Sequence) and not isinstance(value, str):
        if not value:
            raise ValueError("Empty quantity vector")
        return tuple(float(item) for item in value)
    return float(value)


def _unit(quantity: Any) -> str:
    unit = str(quantity.units)
    return "dimensionless" if unit == "dimensionless" else unit


def _path(path: Sequence[str | int]) -> str:
    return ".".join(str(part) for part in path) if path else "<root>"


m = ureg.meter
mm = ureg.millimeter
cm = ureg.centimeter
kg = ureg.kilogram
s = ureg.second
K = ureg.kelvin
N = ureg.newton
Pa = ureg.pascal
GPa = ureg.gigapascal
J = ureg.joule
W = ureg.watt
dimensionless = ureg.dimensionless
