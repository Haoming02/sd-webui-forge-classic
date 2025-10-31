import sys
import os
import cProfile
import pstats
import io

# ensure repo root is on sys.path
repo_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
if repo_root not in sys.path:
    sys.path.insert(0, repo_root)

# monkeypatch main_thread.loop so startup doesn't block forever
import modules_forge.main_thread as mt
mt.loop = lambda: None

# run the launcher start() under cProfile
from modules import launch_utils

if __name__ == '__main__':
    cProfile.run('launch_utils.start()', 'startup.prof')
    s = io.StringIO()
    pstats.Stats('startup.prof', stream=s).sort_stats('cumtime').print_stats(20)
    print(s.getvalue())
