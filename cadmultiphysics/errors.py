from __future__ import annotations

from cadmultiphysics.diagnostics import Diagnostic


class CadMPError(Exception):
    code = "CADMP_ERROR"

    def __init__(self, diagnostics: list[Diagnostic]):
        self.diagnostics = diagnostics
        super().__init__(diagnostics[0].message if diagnostics else self.code)


class SchemaError(CadMPError):
    code = "SCHEMA_ERROR"


class UnitError(CadMPError):
    code = "UNIT_ERROR"


class GeometryError(CadMPError):
    code = "GEOMETRY_ERROR"


class TagError(CadMPError):
    code = "TAG_ERROR"


class MeshError(CadMPError):
    code = "MESH_ERROR"


class DiscretizationError(CadMPError):
    code = "DISCRETIZATION_ERROR"


class FormCompilationError(CadMPError):
    code = "FORM_COMPILATION_ERROR"


class SolverConfigurationError(CadMPError):
    code = "SOLVER_CONFIGURATION_ERROR"


class SolveFailure(CadMPError):
    code = "SOLVE_FAILURE"


class InvariantFailure(CadMPError):
    code = "INVARIANT_FAILURE"


class OutputError(CadMPError):
    code = "OUTPUT_ERROR"


class RestartError(CadMPError):
    code = "RESTART_ERROR"
