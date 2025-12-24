from lib_sve import DecayMethod
from modules import scripts


def grid_reference():
    for data in scripts.scripts_data:
        if data.script_class.__module__ in (
            "scripts.xyz_grid",
            "xyz_grid.py",
        ) and hasattr(data, "module"):
            return data.module

    raise SystemError("Could not find X/Y/Z Plot...")


def xyz_support(cache: dict):

    def apply_field(field):
        def _(p, x, xs):
            cache.update({field: x})

        return _

    xyz_grid = grid_reference()

    extra_axis_options = [
        xyz_grid.AxisOption("SVE Enable", bool, apply_field("enable")),
        xyz_grid.AxisOption("SVE Steps", int, apply_field("steps")),
        xyz_grid.AxisOption("SVE Percentage", float, apply_field("percentage")),
        xyz_grid.AxisOption("SVE Strength", int, apply_field("strength")),
        xyz_grid.AxisOption("SVE Decay", str, apply_field("decay"), choices=DecayMethod.choices()),
        xyz_grid.AxisOption("SVE Clamping", float, apply_field("clamping")),
    ]

    xyz_grid.axis_options.extend(extra_axis_options)
