import math
import unittest

import gradio as gr
from rangeslider import RangeSlider, RangeSliderData


class RangeSliderTest(unittest.TestCase):
    def test_pair_round_trip_and_defaults(self):
        slider = RangeSlider(minimum=0, maximum=1, value=(0.15, 0.85))
        self.assertEqual(slider.value, [0.15, 0.85])
        self.assertEqual(slider.preprocess(slider.postprocess((0.2, 0.8))), (0.2, 0.8))
        self.assertEqual(slider.postprocess(None).root, (0, 1))
        self.assertEqual(slider.postprocess((-1, 2)).root, (0, 1))
        self.assertEqual(slider.step, 0.01)
        self.assertEqual(RangeSlider(value=lambda: (10, 20)).value, [10, 20])

    def test_invalid_bounds_and_values(self):
        for kwargs in ({"minimum": 1, "maximum": 1}, {"minimum": 2, "maximum": 1}, {"maximum": math.inf}, {"step": 0}, {"step": -1}, {"step": math.nan}):
            with self.subTest(kwargs=kwargs), self.assertRaises(ValueError):
                RangeSlider(**kwargs)
        slider = RangeSlider(minimum=0, maximum=1)
        for value in (1, "12", [], [0], [0, 1, 2], [None, 1], [math.nan, 1], [0, math.inf], [0.8, 0.2]):
            with self.subTest(value=value), self.assertRaises(ValueError):
                slider.postprocess(value)
        for value in ((-1, 0.5), (0.5, 2), (0.8, 0.2)):
            with self.subTest(value=value), self.assertRaises(ValueError):
                slider.preprocess(RangeSliderData(root=value))

    def test_component_config_and_events(self):
        with gr.Blocks() as demo:
            slider = RangeSlider(minimum=0, maximum=1, value=(0.2, 0.8), label="Range", visible="hidden")
            slider.change(lambda pair: pair, inputs=slider, outputs=slider)
            slider.input(lambda pair: pair, inputs=slider, outputs=slider)
            slider.release(lambda pair: pair, inputs=slider, outputs=slider)
        self.assertEqual(slider.get_config()["visible"], "hidden")
        self.assertEqual(len(demo.config["dependencies"]), 3)
        self.assertEqual(demo.config["components"][0]["type"], "rangeslider")
        self.assertEqual(slider.api_info()["minItems"], 2)


if __name__ == "__main__":
    unittest.main()
