"""Run frozen July–December test for this method."""
import runpy
import sys
from pathlib import Path

runner = Path(__file__).resolve().parent.parent / 'final_test_2025' / 'test_all.py'
sys.argv = [str(runner), '--method', 'time_series']
runpy.run_path(str(runner), run_name='__main__')
