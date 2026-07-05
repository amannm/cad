from __future__ import annotations

from typing import Annotated, Any, Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator

from cadmultiphysics.diagnostics import Diagnostic
from cadmultiphysics.units import QuantitySpec, UnitSpec

Name = Annotated[str, Field(min_length=1, pattern=r"^[A-Za-z_][A-Za-z0-9_]*$")]
Mode = Literal["linear_steady", "linear_transient", "nonlinear_steady", "nonlinear_transient"]
CellType = Literal["tetrahedron", "hexahedron", "triangle", "quadrilateral", "interval"]
FieldKind = Literal["scalar", "vector"]
EntityType = Literal["box", "cylinder", "boolean_union", "boolean_cut"]
TagNamespace = Literal["materials", "boundaries", "interfaces", "curves", "points"]
StepPhase = Literal["open", "predict", "discretize", "update", "build", "solve", "accept", "commit", "fail", "postprocess", "write"]
OutputCadence = Annotated[str, Field(pattern=r"^(end|every_step|never|every_[1-9][0-9]*)$")]


class StrictModel(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True, populate_by_name=True)


class UnitBlockInput(StrictModel):
    system: Literal["SI"] = "SI"


class GeometryEntityInput(StrictModel):
    type: EntityType
    name: Name
    size: tuple[Any, ...] | None = None
    origin: tuple[Any, ...] | None = None
    radius: Any | None = None
    height: Any | None = None
    axis: tuple[Any, ...] | None = None
    entities: tuple[Name, ...] | None = None
    base: Name | None = None
    tools: tuple[Name, ...] | None = None


class GeometryInput(StrictModel):
    backend: Literal["gmsh_occ"] = "gmsh_occ"
    entities: tuple[GeometryEntityInput, ...]


class SemanticTagInput(StrictModel):
    dim: int = Field(ge=0, le=3)
    entities: tuple[Name, ...] = ()
    selector: str | None = None

    @model_validator(mode="after")
    def complete(self) -> "SemanticTagInput":
        if not self.entities and (self.selector is None or not self.selector.strip()):
            raise ValueError("tag requires entities or selector")
        return self


class TagsInput(StrictModel):
    materials: dict[Name, SemanticTagInput] = Field(default_factory=dict)
    boundaries: dict[Name, SemanticTagInput] = Field(default_factory=dict)
    interfaces: dict[Name, SemanticTagInput] = Field(default_factory=dict)
    curves: dict[Name, SemanticTagInput] = Field(default_factory=dict)
    points: dict[Name, SemanticTagInput] = Field(default_factory=dict)


class ElementInput(StrictModel):
    family: Name
    order: int = Field(ge=0)


class FieldInput(StrictModel):
    name: Name
    kind: FieldKind
    unit: str
    element: ElementInput
    components: int | None = Field(default=None, ge=1)


class MaterialInput(StrictModel):
    model: Name
    parameters: dict[str, Any] = Field(default_factory=dict)


class BoundaryConditionInput(StrictModel):
    name: Name
    on: Name
    field: Name
    type: Name
    value: Any | None = None
    parameters: dict[str, Any] = Field(default_factory=dict)


class LoadInput(StrictModel):
    name: Name
    on: Name
    field: Name
    type: Name
    value: Any | None = None
    parameters: dict[str, Any] = Field(default_factory=dict)


class InitialConditionInput(StrictModel):
    field: Name
    value: Any


class MeshSizeInput(StrictModel):
    global_size: Any = Field(alias="global")
    local: dict[Name, Any] = Field(default_factory=dict)


class MeshInput(StrictModel):
    cell_type: CellType
    order: int = Field(ge=1)
    size: MeshSizeInput
    dimension: int | None = Field(default=None, ge=1, le=3)
    curvature: bool = False
    partitions: int = Field(default=1, ge=1)
    quality: dict[str, Any] = Field(default_factory=dict)


class SolverInput(StrictModel):
    prefix: str = ""
    linear: dict[str, Any] = Field(default_factory=dict)
    nonlinear: dict[str, Any] = Field(default_factory=dict)
    fieldsplits: dict[Name, dict[str, Any]] = Field(default_factory=dict)
    allow_backend_options: bool = False


class OutputInput(StrictModel):
    format: Literal["vtx", "xdmf", "json"] = "vtx"
    fields: tuple[Name, ...] = ()
    derived_fields: tuple[Name, ...] = ()
    cadence: OutputCadence = "end"
    restart_cadence: OutputCadence | None = None
    report_formats: tuple[Literal["json", "csv"], ...] = ("json",)
    writer_options: dict[str, Any] = Field(default_factory=dict)


class TimeInput(StrictModel):
    start: Any
    stop: Any
    step: Any
    scheme: Literal["backward_euler", "crank_nicolson"] = "backward_euler"


class ProblemInput(StrictModel):
    name: Name
    mode: Mode
    version: str = "0.1.0"
    units: UnitBlockInput = Field(default_factory=UnitBlockInput)
    geometry: GeometryInput
    tags: TagsInput
    fields: tuple[FieldInput, ...]
    materials: dict[Name, MaterialInput]
    boundary_conditions: tuple[BoundaryConditionInput, ...] = ()
    loads: tuple[LoadInput, ...] = ()
    initial_conditions: tuple[InitialConditionInput, ...] = ()
    mesh: MeshInput
    solver: SolverInput = Field(default_factory=SolverInput)
    output: OutputInput = Field(default_factory=OutputInput)
    time: TimeInput | None = None
    metadata: dict[str, Any] = Field(default_factory=dict)


class GeometryEntitySpec(StrictModel):
    type: EntityType
    name: str
    size: QuantitySpec | None = None
    origin: QuantitySpec | None = None
    radius: QuantitySpec | None = None
    height: QuantitySpec | None = None
    axis: tuple[float, float, float] | None = None
    entities: tuple[str, ...] = ()
    base: str | None = None
    tools: tuple[str, ...] = ()


class GeometrySpec(StrictModel):
    backend: Literal["gmsh_occ"]
    entities: tuple[GeometryEntitySpec, ...]


class SemanticTagSpec(StrictModel):
    dim: int
    entities: tuple[str, ...] = ()
    selector: str | None = None


class TagsSpec(StrictModel):
    materials: dict[str, SemanticTagSpec]
    boundaries: dict[str, SemanticTagSpec]
    interfaces: dict[str, SemanticTagSpec]
    curves: dict[str, SemanticTagSpec]
    points: dict[str, SemanticTagSpec]


class ElementSpec(StrictModel):
    family: str
    order: int


class FieldSpec(StrictModel):
    name: str
    kind: FieldKind
    components: int
    unit: UnitSpec
    element: ElementSpec


class MaterialSpec(StrictModel):
    model: str
    parameters: dict[str, Any]


class BoundaryConditionSpec(StrictModel):
    name: str
    on: str
    field: str
    type: str
    value: Any | None = None
    parameters: dict[str, Any]


class LoadSpec(StrictModel):
    name: str
    on: str
    field: str
    type: str
    value: Any | None = None
    parameters: dict[str, Any]


class InitialConditionSpec(StrictModel):
    field: str
    value: Any


class MeshSizeSpec(StrictModel):
    global_size: QuantitySpec
    local: dict[str, QuantitySpec]


class MeshPlan(StrictModel):
    cell_type: CellType
    dimension: int
    order: int
    size: MeshSizeSpec
    curvature: bool
    partitions: int
    quality: dict[str, Any]


class PETScProfile(StrictModel):
    prefix: str
    linear: dict[str, Any]
    nonlinear: dict[str, Any]
    fieldsplits: dict[str, dict[str, Any]]
    allow_backend_options: bool


class OutputPlan(StrictModel):
    format: Literal["vtx", "xdmf", "json"]
    fields: tuple[str, ...]
    derived_fields: tuple[str, ...]
    cadence: OutputCadence
    restart_cadence: OutputCadence | None
    report_formats: tuple[str, ...]
    writer_options: dict[str, Any]


class TimePlan(StrictModel):
    start: QuantitySpec
    stop: QuantitySpec
    step: QuantitySpec
    scheme: Literal["backward_euler", "crank_nicolson"]


class ProblemSpec(StrictModel):
    name: str
    version: str
    mode: Mode
    geometry: GeometrySpec
    tags: TagsSpec
    fields: tuple[FieldSpec, ...]
    materials: dict[str, MaterialSpec]
    bcs: tuple[BoundaryConditionSpec, ...]
    loads: tuple[LoadSpec, ...]
    initial_conditions: tuple[InitialConditionSpec, ...]
    mesh: MeshPlan
    solver: PETScProfile
    output: OutputPlan
    time: TimePlan | None
    metadata: dict[str, Any]
    content_hash: str = ""


class DomainIR(StrictModel):
    name: str
    mode: Mode
    entities: tuple[str, ...]
    material_tags: tuple[str, ...]
    boundary_tags: tuple[str, ...]
    fields: tuple[str, ...]
    bcs: tuple[str, ...]
    loads: tuple[str, ...]
    content_hash: str


class GeometryEntityIR(StrictModel):
    name: str
    type: EntityType
    dim: int
    backend_tag: int
    bounds: tuple[float, float, float, float, float, float]


class TagBinding(StrictModel):
    namespace: TagNamespace
    name: str
    dim: int
    entity_tags: tuple[int, ...]
    physical_id: int
    physical_name: str


class TagMap(StrictModel):
    bindings: tuple[TagBinding, ...]


class MeshMetadata(StrictModel):
    backend: Literal["gmsh_occ"]
    format: Literal["msh4"]
    path: str
    dimension: int
    cell_type: CellType
    order: int
    nodes: int
    elements: int
    entities: tuple[GeometryEntityIR, ...]
    tags: TagMap
    physical_groups: dict[str, int]
    physical_names: dict[int, str]
    partition: dict[str, Any]
    quality_report: dict[str, Any] = Field(default_factory=dict)


class FunctionSpacePlan(StrictModel):
    field: str
    kind: FieldKind
    components: int
    element_family: str
    element_order: int
    unit: str
    unit_dimension: str
    block_index: int


class MeasureBindingPlan(StrictModel):
    namespace: TagNamespace
    name: str
    dim: int
    physical_id: int
    physical_name: str


class MaterialBindingPlan(StrictModel):
    material: str
    model: str
    tag: str
    dim: int
    physical_id: int
    entity_tags: tuple[int, ...]


class BoundaryConditionBindingPlan(StrictModel):
    name: str
    type: str
    field: str
    tag: str
    dim: int
    physical_id: int
    value: Any | None = None
    parameters: dict[str, Any] = Field(default_factory=dict)


class LoadBindingPlan(StrictModel):
    name: str
    type: str
    field: str
    tag: str
    dim: int
    physical_id: int
    value: Any | None = None
    parameters: dict[str, Any] = Field(default_factory=dict)


class CouplingDependencyPlan(StrictModel):
    source_field: str
    target_field: str
    mechanism: str


class DiscretePlan(StrictModel):
    content_hash: str = ""
    spec_hash: str
    mesh_hash: str
    mode: Mode
    mesh_dimension: int
    cell_type: CellType
    spaces: tuple[FunctionSpacePlan, ...]
    measures: tuple[MeasureBindingPlan, ...]
    materials: tuple[MaterialBindingPlan, ...]
    boundary_conditions: tuple[BoundaryConditionBindingPlan, ...]
    loads: tuple[LoadBindingPlan, ...]
    coupling: tuple[CouplingDependencyPlan, ...] = ()
    solver_fieldsplits: tuple[str, ...] = ()
    output_fields: tuple[str, ...] = ()


class RunManifest(StrictModel):
    schema_version: str
    content_hash: str
    mesh_options: dict[str, Any]
    solver_options: dict[str, Any]
    output_paths: dict[str, str]
    restart: dict[str, Any]
    artifact_hashes: dict[str, str] = Field(default_factory=dict)


class ModeContract(StrictModel):
    mode: Mode
    problem_kind: Literal["linear", "nonlinear"]
    transient: bool
    solver: Literal["ksp", "snes"]
    physics_interface: Literal["bilinear_linear", "linear_effective_system", "residual_jacobian", "time_discrete_residual_jacobian"]


class TimeGrid(StrictModel):
    start: float
    stop: float
    step: float
    steps: int = Field(ge=1)
    unit: str


class RunPlan(StrictModel):
    mode: Mode
    problem_kind: Literal["linear", "nonlinear"]
    transient: bool
    solver: Literal["ksp", "snes"]
    steps: int = Field(ge=1)
    time: TimeGrid | None = None
    contract: ModeContract
    output_cadence: OutputCadence
    restart_cadence: OutputCadence | None
    output_steps: tuple[int, ...] = ()
    restart_steps: tuple[int, ...] = ()
    checkpoint_policy: Literal["manifest_hash"] = "manifest_hash"
    stop_policy: Literal["fail_closed"] = "fail_closed"


class RunArtifact(StrictModel):
    path: str
    sha256: str


class SolutionState(StrictModel):
    schema_version: str
    content_hash: str
    mode: Mode
    fields: tuple[str, ...]
    committed_step: int = Field(ge=0)
    trial_step: int | None = Field(default=None, ge=1)
    time: float | None = None
    time_unit: str | None = None
    field_state: dict[str, Any] = Field(default_factory=dict)
    history_state: dict[str, Any] = Field(default_factory=dict)
    material_state: dict[str, Any] = Field(default_factory=dict)
    restart_markers: dict[str, str] = Field(default_factory=dict)
    state_hash: str = ""


class AcceptanceCheck(StrictModel):
    name: str
    status: Literal["passed", "failed"]
    diagnostic: str | None = None
    payload: dict[str, Any] = Field(default_factory=dict)


class StepRecord(StrictModel):
    index: int = Field(ge=1)
    status: Literal["accepted", "failed"]
    mode: Mode
    solver: Literal["ksp", "snes"]
    phases: tuple[StepPhase, ...]
    target_time: float | None = None
    time_unit: str | None = None
    opened_state_hash: str
    trial_state_hash: str
    final_state_hash: str
    acceptance: tuple[AcceptanceCheck, ...]
    diagnostics: tuple[Diagnostic, ...] = ()


class RunReport(StrictModel):
    command: Literal["validate", "mesh", "solve", "restart"]
    status: Literal["ok", "failed"]
    name: str | None = None
    mode: Mode | None = None
    content_hash: str | None = None
    accepted_steps: int = Field(default=0, ge=0)
    failed_steps: int = Field(default=0, ge=0)
    artifact_count: int = Field(default=0, ge=0)
    domain: DomainIR | None = None
    run_plan: RunPlan | None = None
    manifest: str | None = None
    manifest_hash: str | None = None
    artifacts: dict[str, RunArtifact] = Field(default_factory=dict)
    state: SolutionState | None = None
    steps: tuple[StepRecord, ...] = ()
    diagnostics: tuple[Diagnostic, ...] = ()


class RestartState(StrictModel):
    schema_version: str
    content_hash: str
    manifest_path: str
    manifest_hash: str
    mode: Mode
    fields: tuple[str, ...]
    step_index: int = Field(ge=0)
    time: QuantitySpec | None = None
    state_hash: str | None = None
    artifacts: dict[str, str] = Field(default_factory=dict)
