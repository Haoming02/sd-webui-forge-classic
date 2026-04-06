from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


@dataclass
class ControlSpec:
    component_type: str
    label: str = ""
    value: Any = None
    minimum: Any = None
    maximum: Any = None
    step: Any = None
    choices: list[Any] = field(default_factory=list)
    elem_id: str | None = None


def normalize_choices(choices) -> list[Any]:
    if choices is None:
        return []

    normalized = []
    for choice in choices:
        normalized.append(choice[0] if isinstance(choice, tuple) else choice)

    return normalized


def control_to_spec(control: Any) -> ControlSpec:
    component_type = getattr(control, "__class__", type(control)).__name__

    return ControlSpec(
        component_type=component_type,
        label=getattr(control, "label", "") or "",
        value=getattr(control, "value", None),
        minimum=getattr(control, "minimum", None),
        maximum=getattr(control, "maximum", None),
        step=getattr(control, "step", None),
        choices=normalize_choices(getattr(control, "choices", None)),
        elem_id=getattr(control, "elem_id", None),
    )


def controls_to_specs(controls: list[Any] | tuple[Any, ...] | None) -> list[ControlSpec]:
    if not controls:
        return []

    return [control_to_spec(control) for control in controls]