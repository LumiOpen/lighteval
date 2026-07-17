"""
name:
MT-Bench Finnish

dataset:
LumiOpen/mtbench_multi

abstract:
Finnish MT-Bench task for the inspect_ai backend. Uses an OpenAI-compatible
scorer model endpoint as the judge.

languages:
finnish

tags:
conversational, generation, multi-turn
"""

from lighteval.tasks.tasks.mt_bench_fi.main_inspect import mt_bench_fi_inspect


TASKS_TABLE = [mt_bench_fi_inspect]
