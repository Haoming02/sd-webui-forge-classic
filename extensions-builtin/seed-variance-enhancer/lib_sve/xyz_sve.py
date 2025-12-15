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

    def choices_decay():
        return ["No decay", "Linear", "Cosine", "Exponential", "Quadratic"]
    
    def choices_decay1():
        return ["No decay", "Linear", "Cosine", "Exponential", "Quadratic"]

    xyz_grid = grid_reference()

    extra_axis_options = [
        xyz_grid.AxisOption("SVE Steps", int, apply_field("steps")),
        xyz_grid.AxisOption("SVE Percentage", float, apply_field("percentage")),
        xyz_grid.AxisOption("SVE Strength", float, apply_field("strength")),
        xyz_grid.AxisOption("SVE Early Decay", str, apply_field("early_decay"), choices=choices_decay1),
        xyz_grid.AxisOption("SVE Mid Threshold", float, apply_field("md_threshold1")),
        xyz_grid.AxisOption("SVE Mid Decay", str, apply_field("mid_decay"), choices=choices_decay),
        xyz_grid.AxisOption("SVE Late Threshold", float, apply_field("threshold2")),
        xyz_grid.AxisOption("SVE Late Decay", str, apply_field("late_decay"), choices=choices_decay),
        
    ]

    xyz_grid.axis_options.extend(extra_axis_options)
