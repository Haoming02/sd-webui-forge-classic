"""Gradio 6 port of freddyaboulton/gradio-range-slider 0.0.8 (Apache-2.0)."""

from __future__ import annotations

import math
from collections.abc import Sequence
from typing import Any, Callable, Literal

from gradio.components.base import FormComponent
from gradio.data_classes import GradioRootModel
from gradio.events import Events
from pydantic import FiniteFloat


class RangeSliderData(GradioRootModel):
    root: tuple[FiniteFloat, FiniteFloat]


class RangeSlider(FormComponent):
    """
    A slider component that allows the user to select a range of values.
    """

    EVENTS = [Events.change, Events.input, Events.release]
    data_model = RangeSliderData
    webui_do_not_create_gradio_pyi_thank_you = True
    FRONTEND_DIR = "./frontend"

    def __init__(
        self,
        minimum: float = 0,
        maximum: float = 100,
        value: tuple[float, float] | Callable | None = None,
        *,
        step: float | None = None,
        label: str | None = None,
        info: str | None = None,
        every: float | None = None,
        show_label: bool | None = None,
        container: bool = True,
        scale: int | None = None,
        min_width: int = 160,
        interactive: bool | None = None,
        visible: bool | Literal["hidden"] = True,
        elem_id: str | None = None,
        elem_classes: list[str] | str | None = None,
        render: bool = True,
    ):
        """
        Parameters:
            minimum: minimum value for slider.
            maximum: maximum value for slider.
            value: default pair. If callable, it is called whenever the app loads.
            step: increment between slider values.
            label: The label for this component. Appears above the component and is also used as the header if there are a table of examples for this component. If None and used in a `gr.Interface`, the label will be the name of the parameter this component is assigned to.
            info: additional component description.
            every: If `value` is a callable, run the function 'every' number of seconds while the client connection is open. Has no effect otherwise. The event can be accessed (e.g. to cancel it) via this component's .load_event attribute.
            show_label: if True, will display label.
            container: If True, will place the component in a container - providing some extra padding around the border.
            scale: relative size compared to adjacent Components. For example if Components A and B are in a Row, and A has scale=2, and B has scale=1, A will be twice as wide as B. Should be an integer. scale applies in Rows, and to top-level Components in Blocks where fill_height=True.
            min_width: minimum pixel width, will wrap if not sufficient screen space to satisfy this value. If a certain scale value results in this Component being narrower than min_width, the min_width parameter will be respected first.
            interactive: if True, slider will be adjustable; if False, adjusting will be disabled. If not provided, this is inferred based on whether the component is used as an input or output.
            visible: If False, component will be hidden.
            elem_id: An optional string that is assigned as the id of this component in the HTML DOM. Can be used for targeting CSS styles.
            elem_classes: An optional list of strings that are assigned as the classes of this component in the HTML DOM. Can be used for targeting CSS styles.
            render: If False, component will not render be rendered in the Blocks context. Should be used if the intention is to assign event listeners now but render the component later.
        """
        if not math.isfinite(minimum) or not math.isfinite(maximum) or minimum >= maximum:
            raise ValueError("RangeSlider minimum must be less than maximum and both must be finite.")
        if step is not None and (not math.isfinite(step) or step <= 0):
            raise ValueError("RangeSlider step must be positive and finite.")

        self.minimum = minimum
        self.maximum = maximum
        if step is None:
            difference = maximum - minimum
            power = math.floor(math.log10(difference) - 2)
            self.step = 10**power
        else:
            self.step = step

        super().__init__(
            label=label,
            info=info,
            every=every,
            show_label=show_label,
            container=container,
            scale=scale,
            min_width=min_width,
            interactive=interactive,
            visible=visible,
            elem_id=elem_id,
            elem_classes=elem_classes,
            render=render,
            value=value,
        )

    def example_payload(self) -> Any:
        return [self.minimum, self.maximum]

    def example_value(self) -> Any:
        return [self.minimum, self.maximum]

    def postprocess(self, value: tuple[float, float] | None) -> RangeSliderData:
        """Validate a pair returned by a callback and clamp it to the slider bounds."""
        if value is None:
            return RangeSliderData(root=(self.minimum, self.maximum))
        if not isinstance(value, Sequence) or isinstance(value, (str, bytes)) or len(value) != 2:
            raise ValueError("Value must be a tuple of two numbers")
        value = RangeSliderData(root=value).root
        if value[0] > value[1]:
            raise ValueError("RangeSlider start must not exceed end.")
        return RangeSliderData(root=(max(self.minimum, min(value[0], self.maximum)), max(self.minimum, min(value[1], self.maximum))))

    def preprocess(self, payload: RangeSliderData) -> tuple[float, float]:
        """Return the validated pair supplied by the browser."""
        start, end = payload.root
        if not self.minimum <= start <= end <= self.maximum:
            raise ValueError("RangeSlider values must be ordered and within the slider bounds.")
        return start, end

