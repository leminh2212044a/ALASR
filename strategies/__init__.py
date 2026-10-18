"""Dang ky chien luoc — run_al.py goi theo TEN."""

from .base import SelectionContext, Strategy
from .random_strategy import (
    RandomStrategy,
    LongestFirstStrategy,
    ShortestFirstStrategy,
)

REGISTRY = {
    "random": RandomStrategy,
    "longest": LongestFirstStrategy,
    "shortest": ShortestFirstStrategy,
    # tuan 5:   "margin", "lconf", "entropy"
    # tuan 6-7: "clustermargin"
    # tuan 8:   "scd", "badge", "coreset"
}


def build(name: str) -> Strategy:
    if name not in REGISTRY:
        raise KeyError(f"chien luoc '{name}' chua co. "
                       f"Dang co: {sorted(REGISTRY)}")
    return REGISTRY[name]()


__all__ = ["SelectionContext", "Strategy", "REGISTRY", "build"]