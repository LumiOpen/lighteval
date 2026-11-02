# Multi-IF

`multiif-en` evaluates the English subset (909
conversations; 909, 909 and 896 scored prompts across the three turns).
`multiif` launches all eight language tasks, with results reported separately
per language. Individual tasks are `multiif-en`, `multiif-zh`, `multiif-fr`,
`multiif-hi`, `multiif-it`, `multiif-pt`, `multiif-ru` and `multiif-es`.

Run with the Inspect backend.
For a standalone smoke test using Inspect's mock model:

```python
from inspect_ai import eval
from lighteval.main_inspect import get_inspect_ai_task
from lighteval.tasks.registry import Registry

config = Registry(tasks="multiif-en").task_to_configs["multiif-en"][0]
eval(get_inspect_ai_task(config), model="mockllm/model", limit=2)
```

Every sample is a complete conversation. Each assistant answer becomes history
for the next user turn. Only existing turns are generated and scored; there is
no synthetic third prompt for shorter conversations. Each turn's cumulative
instruction list is checked against that turn's answer.

`overall` averages the four strict/loose prompt/instruction accuracies within
each turn, then averages across turns. Instruction accuracy pools individual
instructions across conversations within a turn. Per-turn components and
overall scores are also reported. Scores are fractions in [0, 1]. This matches
the historic `show_result.sh` average, rather than averaging per-conversation
instruction fractions.

The dataset is pinned to `facebook/Multi-IF` revision
`0ab97ce0b45c7f57772e8ba2ac1616f4b00bd3aa`. The original instruction checkers
are vendored from `LumiOpen/Multi-IF` commit
`a26c798a4c2a2f123d991b62eb84e27efbc579f5` (`ifeval.py`, Apache-2.0); see
the adjacent LICENSE. They preserve Multi-IF's multilingual counting and
format checks rather than using the different Google IFEval implementation.

Install `lighteval[extended_tasks]` for `langdetect`, `emoji` and `pythainlp`.
The original `langdetect` checks are stochastic on ambiguous text, so rescoring
archived responses can differ slightly from their saved scores.
NLTK's English sentence check also requires `punkt_tab`:

```bash
python -m nltk.downloader punkt_tab
```

The task default is the original 1024-token cap. The original evaluator used
temperature 0.6 and top_p 0.9. Set generation parameters explicitly when
comparing runs. Smoke results demonstrate pipeline operation and should not
be used as benchmark-quality scores.
