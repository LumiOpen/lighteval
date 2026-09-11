"""Execution grader for the HumanEval+/MBPP+ (EvalPlus) tasks.

Grades a model by *running* the code it produced against the benchmark's unit
tests. The public surface deliberately mirrors LiveCodeBench's grader
(``lighteval.tasks.tasks.lcb.codegen_metrics``) so the shared TensorWave network
sandbox (``sandbox_grading.py`` / ``sandboxed_evals.GRADER_PATCHES``) can wrap it
with a single generic wrapper:

  - ``run_test(sample, test=<generation>, timeout=...)`` runs one problem's tests
    and returns ``list[int]`` (``[1]`` pass, ``[0]`` fail, ``[-1]`` sentinel/timeout).
  - ``check_correctness(sample, generation, timeout)`` spawns a
    ``multiprocessing.Process`` running ``run_test`` and reads its result via a
    ``Manager`` proxy — identical control flow to the sandbox wrapper, so behaviour
    is the same whether or not the sandbox monkey-patch is installed.

The ``sample`` dict carries the unit-test harness and entry point; it also carries
a one-element ``input_output`` (``{"inputs": [""], "outputs": [""]}``) purely so the
generic sandbox wrapper's timeout/sentinel math (which indexes
``sample["input_output"]["inputs"]``) works unchanged. Each EvalPlus problem is a
single ``check(entry_point)`` invocation, hence one "input".

``reliability_guard`` is defensive hardening (disables destructive os/shutil/…
builtins), NOT a security boundary — real isolation comes from the network/user
namespace applied by the sandbox wrapper.
"""

import contextlib
import faulthandler
import io
import json
import multiprocessing
import os
import platform
import re
import signal


_SINGLE_INPUT_OUTPUT = json.dumps({"inputs": [""], "outputs": [""]})


def extract_code(model_output: str) -> str:
    """Return the code from the model's (post-reasoning) final answer.

    Prefers the last fenced ``python`` block, then any last fenced block, then
    falls back to the raw text. With ``reasoning_parser=qwen3`` the completion is
    already the final answer (reasoning is separated by vLLM), so this runs on the
    answer text, not the ``<think>`` trace.
    """
    if not model_output:
        return ""
    blocks = re.findall(r"```(?:python|py)?\s*\n(.*?)```", model_output, re.DOTALL)
    if blocks:
        return blocks[-1].strip()
    return model_output.strip()


@contextlib.contextmanager
def _time_limit(seconds: float):
    def _handler(signum, frame):
        raise TimeoutError("timed out")

    signal.setitimer(signal.ITIMER_REAL, seconds)
    signal.signal(signal.SIGALRM, _handler)
    try:
        yield
    finally:
        signal.setitimer(signal.ITIMER_REAL, 0)


@contextlib.contextmanager
def _swallow_io():
    stream = io.StringIO()
    with contextlib.redirect_stdout(stream), contextlib.redirect_stderr(stream):
        yield


_CHECK_DEF = re.compile(r"^\s*def\s+check\s*\(", re.MULTILINE)


def _build_program(sample: dict, generation: str) -> str:
    """Assemble the executable program: model code followed by the test harness.

    The two EvalPlus benchmarks structure their ``test`` field differently:

      - HumanEval+ defines ``def check(candidate): ...`` and must be *invoked*
        as ``check(entry_point)``.
      - MBPP+ has no ``check`` wrapper; its assertions call the target function
        directly at module scope, so they execute as soon as the test runs.

    Appending ``check(entry_point)`` unconditionally raises ``NameError`` for
    every MBPP+ problem (there is no ``check`` to call), scoring the whole task
    0. Only emit the invocation when the harness actually defines ``check``.
    """
    harness = sample.get("test", "")
    entry = sample.get("entry_point", "")
    program = f"{generation}\n\n{harness}\n"
    if entry and _CHECK_DEF.search(harness):
        program += f"\ncheck({entry})\n"
    return program


def _unsafe_execute(program: str, timeout: float) -> bool:
    reliability_guard()
    try:
        exec_globals: dict = {"__name__": "__main__"}
        with _swallow_io():
            with _time_limit(timeout):
                exec(compile(program, "<evalplus>", "exec"), exec_globals)
        return True
    except BaseException:
        return False


def run_test(sample: dict, test=None, timeout: int = 15) -> list:
    """Run one EvalPlus problem. ``test`` is the model-generated code (generation).

    Returns ``[1]`` if the unit tests pass, ``[0]`` otherwise. Signature/return
    shape match lcb's ``run_test`` so the shared sandbox wrapper can drive it.
    """
    generation = test or ""
    program = _build_program(sample, generation)
    return [1 if _unsafe_execute(program, timeout) else 0]


def check_correctness(sample: dict, generation: str, timeout: int) -> list:
    """Run ``run_test`` in a child process; return its ``list[int]`` result.

    Mirrors lcb's ``check_correctness`` (Manager proxy handoff, timeout from
    ``sample["input_output"]["inputs"]``, ``[-1]`` sentinel) so the sandbox
    monkey-patch is a drop-in. Used verbatim when the sandbox is disabled.
    """

    def _temp_run(sample, generation, result):
        result.append(run_test(sample, test=generation, timeout=timeout))

    manager = multiprocessing.Manager()
    result = manager.list()
    p = multiprocessing.Process(target=_temp_run, args=(sample, generation, result))
    p.start()
    p.join(timeout=(timeout + 1) * len(json.loads(sample["input_output"])["inputs"]) + 5)
    if p.is_alive():
        p.kill()
    if not result:
        in_outs = json.loads(sample["input_output"])
        result = [[-1 for _ in range(len(in_outs["inputs"]))]]
    return result[0]


def make_sample(test: str, entry_point: str) -> dict:
    """Build the ``sample`` dict passed to ``check_correctness`` for one problem."""
    return {"input_output": _SINGLE_INPUT_OUTPUT, "test": test, "entry_point": entry_point}


def reliability_guard(maximum_memory_bytes=None):
    """Disable destructive builtins/OS calls before exec'ing untrusted code.

    Vendored from OpenAI human-eval. NOT a security sandbox (its own warning) —
    the real boundary is the CLONE_NEWUSER|CLONE_NEWNET unshare applied per-sample
    by the TensorWave sandbox wrapper.
    """
    import builtins
    import shutil
    import subprocess

    if maximum_memory_bytes is not None:
        import resource

        resource.setrlimit(resource.RLIMIT_AS, (maximum_memory_bytes, maximum_memory_bytes))
        resource.setrlimit(resource.RLIMIT_DATA, (maximum_memory_bytes, maximum_memory_bytes))
        if platform.uname().system != "Darwin":
            resource.setrlimit(resource.RLIMIT_STACK, (maximum_memory_bytes, maximum_memory_bytes))

    faulthandler.disable()

    builtins.exit = None
    builtins.quit = None

    os.environ["OMP_NUM_THREADS"] = "1"

    os.kill = None
    os.system = None
    os.putenv = None
    os.remove = None
    os.removedirs = None
    os.rmdir = None
    os.fchdir = None
    os.setuid = None
    os.fork = None
    os.forkpty = None
    os.killpg = None
    os.rename = None
    os.renames = None
    os.truncate = None
    os.replace = None
    os.unlink = None
    os.fchmod = None
    os.fchown = None
    os.chmod = None
    os.chown = None
    os.chroot = None
    os.lchflags = None
    os.lchmod = None
    os.lchown = None
    os.getcwd = None
    os.chdir = None

    shutil.rmtree = None
    shutil.move = None
    shutil.chown = None

    subprocess.Popen = None

    __builtins__["help"] = None

    import sys

    sys.modules["ipdb"] = None
    sys.modules["joblib"] = None
    sys.modules["resource"] = None
    sys.modules["psutil"] = None
    sys.modules["tkinter"] = None
