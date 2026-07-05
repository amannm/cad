from cadmultiphysics.builder import GeometryBuilder, Problem
from cadmultiphysics.core import build_domain_ir, build_problem_spec, build_run_manifest, build_run_plan, input_json_schema, mode_contract, petsc_options
from cadmultiphysics.schema import DomainIR, MeshMetadata, ModeContract, ProblemInput, ProblemSpec, RestartState, RunManifest, RunPlan, RunReport, SolutionState, StepRecord, TagMap

__version__ = "0.1.0"

__all__ = [
    "__version__",
    "DomainIR",
    "GeometryBuilder",
    "MeshMetadata",
    "ModeContract",
    "Problem",
    "ProblemInput",
    "ProblemSpec",
    "RestartState",
    "RunManifest",
    "RunPlan",
    "RunReport",
    "SolutionState",
    "StepRecord",
    "TagMap",
    "build_domain_ir",
    "build_problem_spec",
    "build_run_manifest",
    "build_run_plan",
    "input_json_schema",
    "mode_contract",
    "petsc_options",
]
