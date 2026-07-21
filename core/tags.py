"""Editable click-to-prompt tag palette loader.

Tags live in ``data/tags.json`` so they can be edited without touching code.
The app renders each group as native ``st.pills`` (Streamlit >= 1.40); selected
pills append their ``text`` to the positive prompt.
"""

from __future__ import annotations

import json
import os
import random
from typing import Any, Dict, List, Optional


def tags_path() -> str:
    root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    return os.path.join(root, "data", "tags.json")


def load_tag_groups(path: str | None = None) -> List[Dict[str, Any]]:
    """Return a list of {'name', 'composable', 'tags': [{'emoji','label','text'}]} groups.

    Falls back to an empty list if the file is missing or malformed, so the UI
    degrades gracefully instead of crashing.
    """
    path = path or tags_path()
    try:
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
    except (FileNotFoundError, json.JSONDecodeError):
        return []

    groups = []
    for group in data.get("groups", []):
        name = group.get("name")
        if not name:
            continue
        tags = []
        for t in group.get("tags", []):
            label = t.get("label")
            if not label:
                continue
            tags.append({
                "emoji": t.get("emoji", ""),
                "label": label,
                "text": t.get("text", label),
            })
        if tags:
            groups.append({
                "name": name,
                "composable": bool(group.get("composable", False)),
                "tags": tags,
            })
    return groups


def tags_to_prompt_fragment(selected_texts: List[str]) -> str:
    """Join selected tag texts into a comma-separated prompt fragment (deduped, ordered)."""
    seen = set()
    ordered = []
    for t in selected_texts:
        stripped = (t or "").strip()
        key = stripped.lower()
        if stripped and key not in seen:
            seen.add(key)
            ordered.append(stripped)
    return ", ".join(ordered)


# Exact-examples-only pluralization map for color+noun composition.
# Only the user-given pairs are special-cased; everything else stays singular.
_COMPOSED_PLURALS: Dict[str, str] = {
    "green eye": "green eyes",
}


def _compose_color_noun(color_text: str, noun_text: str) -> str:
    """Compose a color + noun into a prompt fragment.

    Rules (exact examples only):
      car + blue  -> a blue car
      ball + red  -> a red ball
      tree + yellow -> yellow tree
      eye + green -> green eyes

    Pluralization is handled by an explicit map (_COMPOSED_PLURALS); everything
    else stays in its base form.
    """
    color = (color_text or "").strip()
    noun = (noun_text or "").strip()
    if not color or not noun:
        return f"{color} {noun}".strip()

    # Build the base composed phrase (color first, no leading article).
    base = f"{color} {noun}"
    # Apply exact-examples-only pluralization.
    return _COMPOSED_PLURALS.get(base.lower(), base)


def shuffle_tag_groups(groups: List[Dict[str, Any]], seed: Optional[int] = None) -> List[Dict[str, Any]]:
    """Return a shallow copy of groups with tags shuffled in display order.

    The shuffle is deterministic per session when ``seed`` is provided (or a
    random seed is drawn from ``random``). The group order itself is unchanged;
    only the tag order within each group is shuffled.
    """
    if seed is None:
        seed = random.randint(0, 2**31)
    rng = random.Random(seed)
    shuffled = []
    for g in groups:
        g = dict(g)
        tags = list(g.get("tags", []))
        rng.shuffle(tags)
        g["tags"] = tags
        shuffled.append(g)
    return shuffled
