r"""The editor's worker: jobs computed in a process of its own, one at a time, with the
workbook carried to it as it is; and what happens when the worker fails."""

import os
import pickle
import subprocess
import sys
import threading
import time

import pytest

from relativisticpy import Workbook
from relativisticpy.live import worker as working

KERR = r"g_{\mu\nu} := Kerr(M, a)" + "\nR_{abcd} R^{abcd}"


@pytest.fixture(scope="module")
def vacuum():
    wb = Workbook()
    wb.expr(r"g_{\mu\nu} := Schwarzschild(M)")
    return wb


def answers(computations, job_kind="run", arguments=None, state=None, revision=0):
    """The answers to one job, once it is done."""
    got = []
    done = threading.Event()

    def deliver(kind, data):
        got.append((kind, data))
        done.set()

    job = working.Job(job_kind, arguments or {"text": "x"}, deliver, revision, state)
    computations.submit(job)
    done.wait(60)
    return got


def texts(data):
    return [[value["text"] for value in entry["values"]] for entry in data["statements"]]


def test_a_job_is_done_in_a_worker_with_the_workbook_as_it_was_given(vacuum):
    computations = working.Computations()
    try:
        state = working.portable(vacuum)
        ((kind, data),) = answers(
            computations, arguments={"text": "R_{abcd} R^{abcd}"}, state=state
        )
        assert kind == "ran"
        assert texts(data) == [["48*M^2/r^6"]]
        pid = computations._worker._process.pid
        assert pid != os.getpid()
        ((kind, data),) = answers(
            computations, "compute", {"text": "x^2 + 2x", "start": 0, "end": 3, "action": "compute"}
        )
        assert (kind, data["part"]) == ("computed", "x^2")
        assert computations._worker._process.pid == pid
    finally:
        computations.close()


def test_numbers_made_by_a_worker_never_meet_the_notebooks():
    # A notebook of its own, whose constants are numbered from the start, as a worker's are.
    script = (
        "import threading\n"
        "from relativisticpy import Workbook\n"
        "from relativisticpy.live.worker import Computations, Job, portable\n"
        "wb = Workbook()\n"
        "wb.exe('a := 10; b := 20')\n"
        "got, done = [], threading.Event()\n"
        "def deliver(kind, data):\n"
        "    got.append(data)\n"
        "    done.set()\n"
        "computations = Computations()\n"
        "text = 'c := 30; d := 40; a + b + c + d'\n"
        "computations.submit(Job('run', {'text': text}, deliver, 1, portable(wb)))\n"
        "done.wait(120)\n"
        "print(got[0]['statements'][-1]['values'][0]['text'])\n"
        "computations.close()\n"
    )
    ran = subprocess.run(
        [sys.executable, "-c", script], capture_output=True, text=True, timeout=180, check=False
    )
    assert ran.stdout.strip() == "100", ran.stderr


def test_a_workbook_is_carried_to_another_process_as_it_is(vacuum):
    carried = Workbook._carried(*pickle.loads(working.portable(vacuum)))
    assert carried.expr(r"R_{abcd} R^{abcd}") == vacuum.expr(r"R_{abcd} R^{abcd}")
    assert carried.names().keys() == vacuum.names().keys()


# -- when a worker fails ---------------------------------------------------------------------


class Silent:
    """A worker that starts, takes jobs, and never answers them."""

    def __init__(self, generation, events, memory_limit):
        self.generation = generation
        self.ready = False
        self.events = events
        self.stopped = False

    def start(self):
        self.ready = True

    def send(self, message):
        pass

    def interrupt(self):
        return True

    def stop(self):
        self.stopped = True

    def why(self):
        return None


def test_a_computation_that_does_not_stop_has_its_worker_stopped(monkeypatch):
    monkeypatch.setattr(working, "_Worker", Silent)
    monkeypatch.setattr(working, "GRACE", 0.1)
    computations = working.Computations(time_limit=0.2)
    try:
        ((kind, data),) = answers(computations)
        assert kind == "ran"
        error = data["statements"][0]["error"]
        assert error["message"].startswith("over the time limit of 0.2 s, and it did not stop")
        assert error["hint"].startswith("give the editor longer")
    finally:
        computations.close()


def test_a_worker_that_does_not_start_says_so(monkeypatch):
    class Broken(Silent):
        def start(self):
            raise OSError("no such file")

    monkeypatch.setattr(working, "_Worker", Broken)
    computations = working.Computations()
    try:
        ((kind, data),) = answers(computations, "compute", {"start": 0, "end": 1})
        assert kind == "computed"
        assert data["error"]["message"] == "the editor's worker process did not start"
        assert data["error"]["notes"] == ["no such file"]
    finally:
        computations.close()


def test_a_worker_that_stops_while_computing_fails_that_job_and_the_next_goes_on():
    computations = working.Computations(time_limit=60)
    try:
        assert answers(computations)[0][0] == "ran"
        got = []
        done = threading.Event()

        def deliver(kind, data):
            got.append(data)
            done.set()

        computations.submit(working.Job("run", {"text": KERR}, deliver))
        time.sleep(1.0)
        os.kill(computations._worker._process.pid, 9)
        done.wait(30)
        error = got[0]["statements"][0]["error"]
        assert error["message"] == "RelativisticPy's worker process stopped while computing this"
        assert (
            answers(computations, arguments={"text": "1 + 1"})[0][1]["statements"][0]["values"][0][
                "text"
            ]
            == "2"
        )
    finally:
        computations.close()
