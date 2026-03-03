"""
Scene-graph → natural-language task description utilities.

Converts RLBench scene-graph relationship transitions into language prompts
suitable for VLA policies (GR00T, π0, etc.)  The format is inspired by:

* **ConceptGraphs** (Gu et al., ICRA 2024) — JSON node-list + system prompt
* **DEFAULT_PROMPT** from `build_scenegraph_cfslam.py` — object-relation edges

Only objects and spatial/functional relations that *change* between the
beginning and end of an episode are described; static context is omitted so
the instruction stays concise and action-oriented.

Public API
----------
    scene_graph_transition_to_task(begin_rels, end_rels, objects_meta, task_name)
        → str   (natural-language instruction)
    format_scene_graph_context(objects_meta, relationships)
        → str   (structured scene description for LLM context)
"""

from __future__ import annotations

import json
from typing import Any


# ──────────────────────────────────────────────────────────────────────────
# Relation-type → verb mapping (extend as new RLBench tasks are added)
# ──────────────────────────────────────────────────────────────────────────
_RELATION_VERBS: dict[str, dict[str, str]] = {
    # type → {direction → verb phrase}
    "holding": {
        "acquire": "pick up {obj2}",
        "release": "release {obj2}",
    },
    "stacked": {
        "acquire": "stack {obj1} on {obj2}",
        "release": "unstack {obj1} from {obj2}",
    },
    "on": {
        "acquire": "place {obj1} on {obj2}",
        "release": "remove {obj1} from {obj2}",
    },
    "in": {
        "acquire": "place {obj1} in {obj2}",
        "release": "remove {obj1} from {obj2}",
    },
    "near": {
        "acquire": "move {obj1} near {obj2}",
        "release": "move {obj1} away from {obj2}",
    },
}

# Fallback template when relation type is unknown
_FALLBACK_ACQUIRE = "achieve {type} between {obj1} and {obj2}"
_FALLBACK_RELEASE = "undo {type} between {obj1} and {obj2}"


def _rel_key(rel: dict) -> str:
    """Canonical key for a single relationship dict (order-independent)."""
    return json.dumps(rel, sort_keys=True)


def _pretty_obj(name: str) -> str:
    """Human-readable object name: ``cup_1`` → ``cup 1``."""
    return name.replace("_", " ")


# ──────────────────────────────────────────────────────────────────────────
# Core: transition → language
# ──────────────────────────────────────────────────────────────────────────

def _describe_new_relation(rel: dict) -> str:
    """Describe a *newly acquired* relation as an imperative phrase."""
    rtype = rel.get("type", "unknown")
    obj1 = _pretty_obj(rel.get("object1", "object"))
    obj2 = _pretty_obj(rel.get("object2", "object"))

    verbs = _RELATION_VERBS.get(rtype, {})
    template = verbs.get("acquire", _FALLBACK_ACQUIRE)
    return template.format(obj1=obj1, obj2=obj2, type=rtype)


def _describe_lost_relation(rel: dict) -> str:
    """Describe a *lost* relation as an imperative phrase."""
    rtype = rel.get("type", "unknown")
    obj1 = _pretty_obj(rel.get("object1", "object"))
    obj2 = _pretty_obj(rel.get("object2", "object"))

    verbs = _RELATION_VERBS.get(rtype, {})
    template = verbs.get("release", _FALLBACK_RELEASE)
    return template.format(obj1=obj1, obj2=obj2, type=rtype)


def scene_graph_transition_to_task(
    begin_rels: list[dict],
    end_rels: list[dict],
    objects_meta: dict[str, Any] | None = None,
    task_name: str | None = None,
) -> str:
    """Convert a scene-graph transition into a natural-language task instruction.

    Parameters
    ----------
    begin_rels : list[dict]
        Relationships at the start of the episode segment.
        Each dict has keys ``object1``, ``object2``, ``type``.
    end_rels : list[dict]
        Relationships at the end (goal state) of the episode segment.
    objects_meta : dict, optional
        Object metadata from the scene graph (``scene_graph["objects"]``).
        Currently used only for context, not for the instruction itself.
    task_name : str, optional
        High-level RLBench task name (e.g. ``"stack_cups"``).
        Prepended as context if provided.

    Returns
    -------
    str
        A concise imperative instruction, e.g.
        ``"Pick up cup 1 and stack cup 1 on cup 2."``
    """
    begin_set = {_rel_key(r) for r in begin_rels}
    end_set = {_rel_key(r) for r in end_rels}

    # New relations acquired in this episode
    new_rels = [r for r in end_rels if _rel_key(r) not in begin_set]
    # Relations that disappeared
    lost_rels = [r for r in begin_rels if _rel_key(r) not in end_set]

    parts: list[str] = []
    for r in lost_rels:
        parts.append(_describe_lost_relation(r))
    for r in new_rels:
        parts.append(_describe_new_relation(r))

    if not parts:
        # No change detected — fallback to generic task description
        if task_name:
            return f"Perform the {_pretty_obj(task_name)} task."
        return "Perform the task."

    instruction = ", then ".join(parts) + "."
    # Capitalize first letter
    instruction = instruction[0].upper() + instruction[1:]

    return instruction


def scene_graph_transition_to_task_with_context(
    begin_rels: list[dict],
    end_rels: list[dict],
    objects_meta: dict[str, Any],
    task_name: str | None = None,
) -> str:
    """Like :func:`scene_graph_transition_to_task` but prepends a structured
    scene-graph context block (ConceptGraphs-style) before the instruction.

    The full prompt is of the form::

        Scene: [{"id": "cup_1", "type": "cup_1"}, ...]
        Current relations: [{"object1": "robot", "object2": "cup_1", "type": "holding"}]
        Goal relations: [{"object1": "cup_1", "object2": "cup_2", "type": "stacked"}]
        Instruction: Stack cup 1 on cup 2.

    This richer format is useful when the language encoder has enough capacity
    (e.g. Qwen2.5-7B-Instruct).
    """
    instruction = scene_graph_transition_to_task(
        begin_rels, end_rels, objects_meta, task_name
    )

    scene_lines = format_scene_graph_context(objects_meta, begin_rels)
    goal_lines = json.dumps(end_rels, separators=(",", ":"))

    prompt = (
        f"Scene: {json.dumps(_objects_to_list(objects_meta), separators=(',', ':'))}\n"
        f"Current relations: {json.dumps(begin_rels, separators=(',', ':'))}\n"
        f"Goal relations: {goal_lines}\n"
        f"Instruction: {instruction}"
    )
    return prompt


# ──────────────────────────────────────────────────────────────────────────
# Helpers for structured scene description (ConceptGraphs-inspired)
# ──────────────────────────────────────────────────────────────────────────

def _objects_to_list(objects_meta: dict[str, Any]) -> list[dict]:
    """Convert the objects dict from scene_graph.json to a JSON-serialisable list."""
    result = []
    for obj_id, meta in objects_meta.items():
        entry = {"id": obj_id}
        if isinstance(meta, dict):
            entry.update(meta)
        result.append(entry)
    return result


def format_scene_graph_context(
    objects_meta: dict[str, Any],
    relationships: list[dict],
) -> str:
    """Format a scene-graph snapshot into a concise text block.

    Parameters
    ----------
    objects_meta : dict
        ``scene_graph["objects"]`` mapping.
    relationships : list[dict]
        Current relationship list.

    Returns
    -------
    str
        Multi-line text suitable for prepending to an LLM prompt.
    """
    obj_list = _objects_to_list(objects_meta)
    lines = [
        f"Objects: {json.dumps(obj_list, separators=(',', ':'))}",
        f"Relations: {json.dumps(relationships, separators=(',', ':'))}",
    ]
    return "\n".join(lines)


# ──────────────────────────────────────────────────────────────────────────
# Batch helper: generate all task descriptions for a set of episodes
# ──────────────────────────────────────────────────────────────────────────

def generate_task_descriptions(
    episodes: list[dict],
    objects_meta: dict[str, Any],
    task_name: str,
    *,
    use_context: bool = False,
) -> list[str]:
    """Generate task descriptions for every episode in a converted dataset.

    Parameters
    ----------
    episodes : list[dict]
        Each dict must have ``begin_sg`` and ``end_sg`` keys (lists of
        relationship dicts).
    objects_meta : dict
        Object metadata from the scene graph.
    task_name : str
        RLBench task name.
    use_context : bool
        If True, use the richer ConceptGraphs-style prompt that includes the
        full scene description.  Default False (concise imperative only).

    Returns
    -------
    list[str]
        One task description string per episode.
    """
    fn = (
        scene_graph_transition_to_task_with_context
        if use_context
        else scene_graph_transition_to_task
    )
    descs = []
    for ep in episodes:
        if use_context:
            desc = fn(ep["begin_sg"], ep["end_sg"], objects_meta, task_name)
        else:
            desc = fn(ep["begin_sg"], ep["end_sg"], objects_meta, task_name)
        descs.append(desc)
    return descs
