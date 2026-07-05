from __future__ import annotations

from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field


class Diagnostic(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    code: str
    severity: Literal["error", "warning", "info"] = "error"
    message: str
    path: tuple[str | int, ...] = ()
    step_index: int | None = None
    source: str
    backend_error: str | None = None
    hint: str | None = None
    payload: dict[str, Any] = Field(default_factory=dict)


def schema_diagnostics(exc: Exception) -> list[Diagnostic]:
    errors = getattr(exc, "errors", lambda: [])()
    if not errors:
        return [
            Diagnostic(
                code="SCHEMA_VALIDATION_FAILED",
                message=str(exc),
                source="schema",
            )
        ]
    return [
        Diagnostic(
            code=f"SCHEMA_{str(error.get('type', 'invalid')).upper()}",
            message=str(error.get("msg", "Invalid input")),
            path=tuple(error.get("loc", ())),
            source="schema",
            payload={"input": error.get("input")} if "input" in error else {},
        )
        for error in errors
    ]


def diagnostics_payload(status: str, diagnostics: list[Diagnostic]) -> dict[str, Any]:
    return {
        "status": status,
        "diagnostics": [diagnostic.model_dump(mode="json") for diagnostic in diagnostics],
    }
