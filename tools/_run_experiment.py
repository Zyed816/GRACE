from pathlib import Path
import runpy


def run(relative_target: str):
    target = Path(__file__).resolve().parents[1] / relative_target
    runpy.run_path(str(target), run_name="__main__")
