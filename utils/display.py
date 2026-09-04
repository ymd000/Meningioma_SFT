"""display.py — 色・順序・略称の表示ヘルパー共通処理。"""
from __future__ import annotations

import matplotlib.pyplot as plt


def ordered_subtypes(present: set, cfg: dict) -> list[str]:
    config_order = list(cfg.get("display", {}).get("colors", {}).get("subtypes", {}).keys())
    ordered = [s for s in config_order if s in present]
    remaining = sorted(s for s in present if s not in config_order)
    return ordered + remaining


def subtype_color_map(subtypes: list[str], cfg: dict) -> dict[str, str]:
    colors = cfg.get("display", {}).get("colors", {})
    explicit = colors.get("subtypes", {})
    sft_color = colors.get("sft", "#000000")
    tab20 = plt.get_cmap("tab20")
    auto_i, result = 0, {}
    for sub in subtypes:
        if "sft" in sub.lower():
            result[sub] = sft_color
        elif sub in explicit:
            result[sub] = explicit[sub]
        else:
            result[sub] = tab20(auto_i % 20)
            auto_i += 1
    return result


def shorten(name: str, cfg: dict) -> str:
    return cfg.get("display", {}).get("shortened", {}).get("subtypes", {}).get(name, name)


def src_code(src: str) -> str:
    return "".join(p[0].upper() for p in src.replace("-", "_").split("_") if p)[:3]


def make_abbrev(labels: list[str], cfg: dict) -> dict[str, str]:
    shortened = cfg.get("display", {}).get("shortened", {}).get("subtypes", {})
    result = {}
    for label in labels:
        if "__" in label:
            s, sub = label.split("__", 1)
            result[label] = f"{src_code(s)}-{shortened.get(sub, sub[:4].capitalize())}"
        else:
            result[label] = shortened.get(label, label[:4].capitalize())
    return result
