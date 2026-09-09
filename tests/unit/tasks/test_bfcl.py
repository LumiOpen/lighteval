"""Unit tests for the BFCL non-live grading core.

Skipped when the optional ``bfcl-eval`` dependency is not installed.
"""

import pytest


pytest.importorskip("bfcl_eval", reason="BFCL tasks require the 'bfcl-eval' package")

from lighteval.tasks.tasks.bfcl import core  # noqa: E402


# A simple_python-style function schema + ground truth (BFCL format: each arg maps
# to a list of acceptable values; "" in the list means the arg may be omitted).
FUNCTION = [
    {
        "name": "calculate_triangle_area",
        "description": "Calculate the area of a triangle given its base and height.",
        "parameters": {
            "type": "dict",
            "properties": {
                "base": {"type": "integer", "description": "The base."},
                "height": {"type": "integer", "description": "The height."},
                "unit": {"type": "string", "description": "Unit (default 'units')."},
            },
            "required": ["base", "height"],
        },
    }
]
GROUND_TRUTH = [{"calculate_triangle_area": {"base": [10], "height": [5], "unit": ["units", ""]}}]
CALL = "[calculate_triangle_area(base=10, height=5)]"


@pytest.mark.parametrize(
    ("text", "expected"),
    [
        (CALL, 1.0),  # clean correct call
        (f"[wrong_{CALL[1:]}", 0.0),  # wrong function name
        (f"<think>reasoning</think>\n{CALL}", 1.0),  # <think> stripped
        (f"Reasoning trace:\n{CALL}", 1.0),  # reasoning-prefix leak
        (f"Reasoning trace:\nlots of prose\n\nResponse:\n\n{CALL}", 1.0),  # Response: section
        ('[{"name": "calculate_triangle_area", "params": {"base": 10}}]', 0.0),  # JSON rejected
        ("I cannot help with that.", 0.0),  # no decodable call
    ],
)
def test_simple_python_grading(text, expected):
    score, _ = core.grade(text, "simple_python", FUNCTION, GROUND_TRUTH)
    assert score == expected


def test_irrelevance_success_when_no_call():
    # non-live `irrelevance` and live `live_irrelevance`: correct iff NO call emitted
    for cat in ("irrelevance", "live_irrelevance"):
        assert core.grade("Sorry, no tool fits.", cat, FUNCTION, None)[0] == 1.0
        assert core.grade(CALL, cat, FUNCTION, None)[0] == 0.0


def test_live_relevance_success_when_call_emitted():
    # live `live_relevance`: correct iff a call IS emitted (opposite of irrelevance)
    assert core.grade(CALL, "live_relevance", FUNCTION, None)[0] == 1.0
    assert core.grade("Sorry, no tool fits.", "live_relevance", FUNCTION, None)[0] == 0.0


def test_parallel_requires_all_calls():
    functions = [
        {
            "name": "play",
            "description": "Play a song.",
            "parameters": {
                "type": "dict",
                "properties": {
                    "artist": {"type": "string", "description": "Artist."},
                    "duration": {"type": "integer", "description": "Minutes."},
                },
                "required": ["artist", "duration"],
            },
        }
    ]
    gt = [
        {"play": {"artist": ["Taylor Swift"], "duration": [20]}},
        {"play": {"artist": ["Maroon 5"], "duration": [15]}},
    ]
    both = "[play(artist='Taylor Swift', duration=20), play(artist='Maroon 5', duration=15)]"
    one = "[play(artist='Taylor Swift', duration=20)]"
    assert core.grade(both, "parallel", functions, gt)[0] == 1.0
    assert core.grade(one, "parallel", functions, gt)[0] == 0.0
