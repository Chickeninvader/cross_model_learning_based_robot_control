"""
Scene-graph -> task-description utilities.

Task description format is intentionally simple and structured:

    object1 relation object2, object3 relation object4

Descriptions are generated from scene-graph transitions, prioritizing goal
relationships (end state) that differ from the beginning state.
"""

from __future__ import annotations

import json
from typing import Any


def _rel_key(rel: dict) -> str:
    """Canonical key for a single relationship dict (order-independent)."""
    return json.dumps(rel, sort_keys=True)


def _pretty_obj(name: str) -> str:
    """Keep object id stable for training prompts."""
    return str(name)


def _object_phrase(obj_id: str, objects_meta: dict[str, Any] | None) -> str:
    """Resolve object id to an attribute-aware phrase (e.g. 'white rubbish')."""
    if not objects_meta:
        return _pretty_obj(obj_id)

    meta = objects_meta.get(obj_id, {})
    if not isinstance(meta, dict):
        return _pretty_obj(obj_id)

    base_phrase = _pretty_obj(obj_id)
    base_type = meta.get("type")
    if isinstance(base_type, str) and base_type.strip():
        base_phrase = _pretty_obj(base_type.strip())
    else:
        # Fallback to display name when type is unavailable.
        display_name = meta.get("display_name")
        if isinstance(display_name, str) and display_name.strip():
            base_phrase = _pretty_obj(display_name.strip())

    attrs = meta.get("attributes", {})
    color = attrs.get("color") if isinstance(attrs, dict) else None
    if isinstance(color, str) and color.strip() and color.strip().lower() != "unknown":
        return f"{color.strip()} {base_phrase}".strip()
    return base_phrase


def _relation_triplet(
    rel: dict,
    objects_meta: dict[str, Any] | None = None,
) -> str:
    """Format one relation triplet: 'obj1 relation obj2'."""
    obj1 = _object_phrase(rel.get("object1", "object"), objects_meta)
    rel_type = str(rel.get("type", "related_to"))
    obj2 = _object_phrase(rel.get("object2", "object"), objects_meta)
    return f"{obj1} {rel_type} {obj2}"


def _sorted_unique_relations(rels: list[dict]) -> list[dict]:
    """Deduplicate and return stable sorted relation dicts."""
    uniq: dict[str, dict] = {}
    for r in rels:
        uniq[_rel_key(r)] = r
    vals = list(uniq.values())
    vals.sort(
        key=lambda r: (
            str(r.get("object1", "")),
            str(r.get("type", "")),
            str(r.get("object2", "")),
        )
    )
    return vals


def scene_graph_transition_to_task(
    begin_rels: list[dict],
    end_rels: list[dict],
    objects_meta: dict[str, Any] | None = None,
    task_name: str | None = None,
) -> str:
    """Convert a scene-graph transition into a relation-triplet task string.

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
        Comma-separated relation triplets, e.g.
        ``"robot holding cup_1, cup_1 on cup_2"``
    """
    begin_set = {_rel_key(r) for r in begin_rels}
    end_set = {_rel_key(r) for r in end_rels}

    # Prefer goal-side transition edges (new in end state).
    goal_delta = [r for r in end_rels if _rel_key(r) not in begin_set]
    goal_delta = _sorted_unique_relations(goal_delta)

    # If there are no new goal edges, fallback to full end state.
    if not goal_delta:
        goal_delta = _sorted_unique_relations(end_rels)

    # If end state is empty, fallback to relations removed from begin state.
    if not goal_delta:
        removed = [r for r in begin_rels if _rel_key(r) not in end_set]
        goal_delta = _sorted_unique_relations(removed)

    if not goal_delta:
        if task_name:
            return _pretty_obj(task_name)
        return "task"

    parts = [_relation_triplet(r, objects_meta) for r in goal_delta]
    return ", ".join(parts)


def scene_graph_goal_to_task(
    end_rels: list[dict],
    objects_meta: dict[str, Any] | None = None,
) -> str:
    """Generate task string from goal scene graph only.

    Output format: ``obj relation obj, obj relation obj``.
    """
    goal_rels = _sorted_unique_relations(end_rels or [])
    if not goal_rels:
        return "task"
    return ", ".join(_relation_triplet(r, objects_meta) for r in goal_rels)


def scene_graph_transition_to_task_with_context(
    begin_rels: list[dict],
    end_rels: list[dict],
    objects_meta: dict[str, Any],
    task_name: str | None = None,
) -> str:
    """Like :func:`scene_graph_transition_to_task` but prepends a structured
    scene-graph context block (ConceptGraphs-style) before the instruction.

    The full prompt includes:
    1. A description of the prompt format
    2. An example showing the structure
    3. The actual scene graph transition data

    This richer format is useful when the language encoder has enough capacity
    (e.g. Qwen2.5-7B-Instruct).
    """
    instruction = scene_graph_transition_to_task(
        begin_rels, end_rels, objects_meta, task_name
    )

    # Build the actual scene data
    scene_json = json.dumps(_objects_to_list(objects_meta), separators=(',', ':'))
    current_rels_json = json.dumps(begin_rels, separators=(',', ':'))
    goal_rels_json = json.dumps(end_rels, separators=(',', ':'))

    # Construct the full prompt with description, example, and actual data
    prompt = (
        "This prompt describes a robotic manipulation task using scene graphs. "
        "It consists of a structured scene description followed by an instruction. "
        "The scene includes objects with their metadata and relationships between them. "
        "The instruction describes changes needed to transition from the current state to the goal state.\n"
        "\n"
        "Example format:\n"
        'Scene: [{"id":"cup_1","type":"cup"},{"id":"cup_2","type":"cup"},{"id":"robot","type":"robot"}]\n'
        'Current relations: [{"object1":"robot","object2":"cup_1","type":"holding"}]\n'
        'Goal relations: [{"object1":"cup_1","object2":"cup_2","type":"stacked"}]\n'
        "Instruction: Stack cup 1 on cup 2.\n"
        "\n"
        "Current task:\n"
        f"Scene: {scene_json}\n"
        f"Current relations: {current_rels_json}\n"
        f"Goal relations: {goal_rels_json}\n"
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
    objects_meta: dict[str, Any] | None,
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
        RLBench task name (currently unused for non-context mode).
    use_context : bool
        If True, use the richer ConceptGraphs-style prompt that includes the
        full scene description.  Default False (concise imperative only).

    Returns
    -------
    list[str]
        One task description string per episode.
    """
    descs = []
    for ep in episodes:
        begin_sg = ep.get("begin_sg", [])
        end_sg = ep.get("end_sg", [])
        if use_context:
            # Context mode should include transition information (begin -> end).
            desc = scene_graph_transition_to_task_with_context(
                begin_sg,
                end_sg,
                objects_meta or {},
                task_name=None,
            )
        else:
            # Non-context mode should use goal scene graph only.
            desc = scene_graph_goal_to_task(end_sg, objects_meta)
        descs.append(desc)
    return descs
