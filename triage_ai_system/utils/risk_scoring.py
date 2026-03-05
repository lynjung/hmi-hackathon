"""Utility helpers for urgency color-coding and display formatting."""

from __future__ import annotations


def urgency_color(level: str) -> str:
    mapping = {
        "Emergency": "#ff4b4b",
        "Urgent": "#ffa500",
        "Moderate": "#f4d03f",
        "Low": "#2ecc71",
    }
    return mapping.get(level, "#95a5a6")
