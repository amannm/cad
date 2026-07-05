from cadmultiphysics.core import build_domain_ir, build_problem_spec, build_run_manifest, input_json_schema
from cadmultiphysics.schema import DomainIR, ProblemInput, ProblemSpec, RunManifest

__version__ = "0.1.0"

__all__ = [
    "__version__",
    "DomainIR",
    "ProblemInput",
    "ProblemSpec",
    "RunManifest",
    "build_domain_ir",
    "build_problem_spec",
    "build_run_manifest",
    "input_json_schema",
]
