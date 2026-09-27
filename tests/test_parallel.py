import json
import subprocess
import sys

import pytest

from speaker_benchmark.runner import run, run_workers


def test_parallel_workers_overlap_and_respect_capacity(tmp_path):
    script = tmp_path / "worker.py"
    script.write_text('''import sys, time
from pathlib import Path
root, index = Path(sys.argv[1]), int(sys.argv[2])
(root / f"started-{index}").touch()
if index < 2:
    deadline = time.monotonic() + 5
    while not all((root / f"started-{i}").exists() for i in range(2)):
        if time.monotonic() > deadline:
            sys.exit(3)
        time.sleep(.01)
    assert not (root / "started-2").exists()
    (root / f"finished-{index}").touch()
else:
    assert any((root / f"finished-{i}").exists() for i in range(2))
''')
    jobs = []
    for index in range(3):
        directory = tmp_path / str(index)
        directory.mkdir()
        jobs.append((directory, [sys.executable, str(script), str(tmp_path), str(index)]))
    run_workers(jobs, 2, 10)
    for directory, _ in jobs:
        state = json.loads((directory / "worker.json").read_text())
        assert state["exit_code"] == 0 and not state["timed_out"]


def test_timeout_is_recorded_and_next_job_still_runs(tmp_path):
    jobs = []
    for index, code in enumerate(["import time; time.sleep(60)", "pass"]):
        directory = tmp_path / str(index)
        directory.mkdir()
        jobs.append((directory, [sys.executable, "-c", code]))
    run_workers(jobs, 1, .15)
    assert json.loads((tmp_path / "0/worker.json").read_text())["timed_out"]
    assert json.loads((tmp_path / "1/worker.json").read_text())["exit_code"] == 0


def test_launch_failure_cleans_up_existing_workers(tmp_path, monkeypatch):
    popen = subprocess.Popen
    processes = []

    def launch(*args, **kwargs):
        if processes:
            raise OSError("fixture launch failure")
        process = popen(*args, **kwargs)
        processes.append(process)
        return process

    monkeypatch.setattr("speaker_benchmark.runner.subprocess.Popen", launch)
    job = (tmp_path, [sys.executable, "-c", "import time; time.sleep(60)"])
    with pytest.raises(OSError, match="fixture"):
        run_workers([job, job], 2, 10)
    assert processes[0].poll() is not None


@pytest.mark.parametrize("parallel", [0, -1, True, 1.5])
def test_invalid_parallelism_is_rejected_before_loading_data(parallel):
    with pytest.raises(ValueError, match="positive integer"):
        run("unused", "unused", "unused", parallel)
