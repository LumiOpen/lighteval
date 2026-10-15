"""Include Base 44 multiple-choice evaluation for Finnish.

The prompt and zero-shot accuracy match lm-eval's include_base_44_finnish
group. Its five subject categories are all in the 551-row Finnish test split.
"""

from lighteval.metrics.metrics import Metrics
from lighteval.tasks.lighteval_task import LightevalTaskConfig
from lighteval.tasks.requests import Doc


def include_base_44_finnish_prompt(line: dict, task_name: str) -> Doc:
    query = (
        f"{line['question'].strip()}\n"
        f"A. {line['option_a']}\n"
        f"B. {line['option_b']}\n"
        f"C. {line['option_c']}\n"
        f"D. {line['option_d']}\n"
        "Answer:"
    )
    return Doc(
        task_name=task_name,
        query=query,
        choices=["A", "B", "C", "D"],
        gold_index=int(line["answer"]),
    )


TASKS_TABLE = [
    LightevalTaskConfig(
        name="include_base_44_finnish",
        prompt_function=include_base_44_finnish_prompt,
        hf_repo="CohereLabs/include-base-44",
        hf_subset="Finnish",
        hf_avail_splits=["test"],
        evaluation_splits=["test"],
        metrics=[Metrics.loglikelihood_acc],
        version=0,
    ),
]
