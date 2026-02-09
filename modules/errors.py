import sys
import textwrap
import traceback

exception_records = []


def format_traceback(tb):
    return [[f"{x.filename}, line {x.lineno}, {x.name}", x.line] for x in traceback.extract_tb(tb)]


def format_exception(e, tb):
    return {"exception": str(e), "traceback": format_traceback(tb)}


def get_exceptions():
    try:
        return list(reversed(exception_records))
    except Exception as e:
        return str(e)


def record_exception():
    _, e, tb = sys.exc_info()
    if e is None:
        return

    if exception_records and exception_records[-1] == e:
        return

    exception_records.append(format_exception(e, tb))

    if len(exception_records) > 5:
        exception_records.pop(0)


def report(message: str, *, exc_info: bool = False) -> None:
    """
    Print an error message to stderr, with optional traceback.
    """

    record_exception()

    for line in message.splitlines():
        print("***", line, file=sys.stderr)
    if exc_info:
        print(textwrap.indent(traceback.format_exc(), "    "), file=sys.stderr)
        print("---", file=sys.stderr)


def print_error_explanation(message):
    record_exception()

    lines = message.strip().split("\n")
    max_len = max([len(x) for x in lines])

    print("=" * max_len, file=sys.stderr)
    for line in lines:
        print(line, file=sys.stderr)
    print("=" * max_len, file=sys.stderr)


def display(e: Exception, task, *, full_traceback=False):
    record_exception()

    print(f"{task or 'error'}: {type(e).__name__}", file=sys.stderr)
    te = traceback.TracebackException.from_exception(e)
    if full_traceback:
        # include frames leading up to the try-catch block
        te.stack = traceback.StackSummary(traceback.extract_stack()[:-2] + te.stack)
    print(*te.format(), sep="", file=sys.stderr)


already_displayed = {}


def display_once(e: Exception, task):
    record_exception()

    if task in already_displayed:
        return

    display(e, task)

    already_displayed[task] = 1


def run(code, task):
    try:
        code()
    except Exception as e:
        display(task, e)


def check_versions():
    import gradio
    import torch
    from packaging import version

    from modules import shared

    expected_torch = "2.10.0"
    expected_xformers = "0.0.34"
    expected_gradio = "4.39.0"

    _outdated = False

    if version.parse(torch.__version__) < version.parse(expected_torch):
        _outdated = True
        print_error_explanation(
            f"""
            You are running torch {torch.__version__}, which is outdated.
            To install the latest version, run with commandline flag --reinstall-torch.
            """.strip()
        )

    if shared.xformers_available:
        import xformers

        if version.parse(xformers.__version__) < version.parse(expected_xformers):
            _outdated = True
            print_error_explanation(
                f"""
                You are running xformers {xformers.__version__}, which is outdated.
                To install the latest version, run with commandline flag --reinstall-xformers.
                """.strip()
            )

    if version.parse(gradio.__version__) < version.parse(expected_gradio):
        _outdated = True
        print_error_explanation(
            f"""
            You are running gradio {gradio.__version__}.
            This program was built on gradio {expected_gradio}.
            Using a different version of gradio is likely to break the program.
            """.strip()
        )

    if _outdated:
        print("\nUse --skip-version-check commandline argument to disable the version check(s).\n")
