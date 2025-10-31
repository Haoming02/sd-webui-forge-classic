import sys
import os
import cProfile
import pstats
import io

# ensure repository root is on sys.path so we can import local modules
repo_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
if repo_root not in sys.path:
    sys.path.insert(0, repo_root)

import modules_forge.main_thread as mt


def busy():
    # CPU-bound workload to exercise the main_thread
    return sum(i * i for i in range(2000000))


def run():
    mt.start()
    mt.run_and_wait_result(busy)
    mt.run_and_wait_result(busy)
    mt.stop()


if __name__ == '__main__':
    cProfile.run('run()', 'small_ui.prof')
    s = io.StringIO()
    pstats.Stats('small_ui.prof', stream=s).sort_stats('cumtime').print_stats(20)
    print(s.getvalue())
