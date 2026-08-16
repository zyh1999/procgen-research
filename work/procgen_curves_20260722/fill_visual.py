#!/usr/bin/env python3
from pathlib import Path

template = Path(
    "/Users/user/.codex/visualizations/2026/07/20/019f8055-8ac8-7b41-a529-37dbaa4704aa/procgen-rat-vs-ppo-curves.html"
)
data = Path("/Users/user/Documents/procgen/work/procgen_curves_20260722/curve_data.json")
text = template.read_text()
if text.count("__CURVE_DATA__") != 1:
    raise RuntimeError("Expected one data placeholder")
template.write_text(text.replace("__CURVE_DATA__", data.read_text()))
