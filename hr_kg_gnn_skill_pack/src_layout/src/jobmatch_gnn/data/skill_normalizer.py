"""Skill normalization utilities."""
from __future__ import annotations


def normalize_skill(skill: str, alias_map: dict[str, str] | None = None) -> str:
    """Return normalized skill name."""
    key = skill.strip().lower()
    if alias_map and key in alias_map:
        return alias_map[key]
    return skill.strip()


def normalize_skills(skills: list[str], alias_map: dict[str, str] | None = None) -> list[str]:
    """Normalize and deduplicate skill names while preserving order."""
    seen: set[str] = set()
    output: list[str] = []
    for skill in skills:
        normalized = normalize_skill(skill, alias_map)
        if normalized and normalized not in seen:
            seen.add(normalized)
            output.append(normalized)
    return output
