#!/usr/bin/env python3
from __future__ import annotations

import csv
import json
import math
import re
from collections import defaultdict
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw, ImageFont


ROOT = Path(__file__).resolve().parent
RAW = ROOT / "raw"
OUT = Path("/Users/user/Documents/procgen/outputs")

ENV_ORDER = [
    "bigfish-easy-0-10",
    "bossfight-easy-0-10",
    "caveflyer-easy-0-10",
    "coinrun-easy-0-10",
    "jumper-easy-0-10",
    "maze-easy-0-10",
    "miner-easy-0-10",
    "starpilot-easy-0-10",
]

ENV_LABEL = {
    "bigfish-easy-0-10": "BigFish",
    "bossfight-easy-0-10": "BossFight",
    "caveflyer-easy-0-10": "CaveFlyer",
    "coinrun-easy-0-10": "CoinRun",
    "jumper-easy-0-10": "Jumper",
    "maze-easy-0-10": "Maze",
    "miner-easy-0-10": "Miner",
    "starpilot-easy-0-10": "StarPilot",
}

METHOD_DIRS = {
    "PPO": [RAW / "ppo54" / "runs", RAW / "ppo92" / "runs"],
    "Exact RAT": [
        RAW / "rat54" / "runs",
        RAW / "rat92a" / "runs",
        RAW / "rat92b" / "runs",
    ],
}

ROW_RE = re.compile(r"^\|\s*([^|]+?)\s*\|\s*([^|]+?)\s*\|\s*$")


def parse_stdout(path: Path) -> tuple[np.ndarray, np.ndarray]:
    points: list[tuple[float, float]] = []
    block: dict[str, str] = {}
    for line in path.read_text(errors="replace").splitlines():
        match = ROW_RE.match(line)
        if match:
            block[match.group(1).strip()] = match.group(2).strip()
            continue
        if line.startswith("---") and block:
            if "misc/total_timesteps" in block and "eprewmean" in block:
                points.append(
                    (
                        float(block["misc/total_timesteps"]),
                        float(block["eprewmean"]),
                    )
                )
            block = {}
    if block and "misc/total_timesteps" in block and "eprewmean" in block:
        points.append(
            (float(block["misc/total_timesteps"]), float(block["eprewmean"]))
        )
    if not points:
        raise RuntimeError(f"No curve points parsed from {path}")
    dedup: dict[float, float] = {}
    for step, value in points:
        dedup[step] = value
    steps = np.asarray(sorted(dedup), dtype=float)
    values = np.asarray([dedup[s] for s in steps], dtype=float)
    return steps, values


def collect() -> dict[str, dict[str, dict[int, tuple[np.ndarray, np.ndarray]]]]:
    curves: dict[str, dict[str, dict[int, tuple[np.ndarray, np.ndarray]]]] = {
        method: defaultdict(dict) for method in METHOD_DIRS
    }
    for method, roots in METHOD_DIRS.items():
        for root in roots:
            if not root.exists():
                continue
            for stdout in sorted(root.glob("*/seed*/stdout.log")):
                env = stdout.parent.parent.name
                seed = int(stdout.parent.name.removeprefix("seed"))
                rc_path = stdout.parent / "returncode"
                if not rc_path.exists() or rc_path.read_text().strip() != "0":
                    raise RuntimeError(f"Nonzero or missing return code: {stdout}")
                if seed in curves[method][env]:
                    raise RuntimeError(f"Duplicate {method}/{env}/seed{seed}")
                curves[method][env][seed] = parse_stdout(stdout)
    for method in METHOD_DIRS:
        for env in ENV_ORDER:
            seeds = curves[method].get(env, {})
            if sorted(seeds) != [0, 1, 2, 3, 4]:
                raise RuntimeError(
                    f"Coverage mismatch for {method}/{env}: {sorted(seeds)}"
                )
    return curves


def aggregate(curves):
    aggregated = {}
    for env in ENV_ORDER:
        aggregated[env] = {}
        for method in METHOD_DIRS:
            seed_curves = curves[method][env]
            common_steps = sorted(
                set.intersection(*(set(v[0].tolist()) for v in seed_curves.values()))
            )
            if not common_steps:
                raise RuntimeError(f"No common grid for {method}/{env}")
            steps = np.asarray(common_steps, dtype=float)
            rows = []
            for seed in range(5):
                seed_steps, seed_values = seed_curves[seed]
                lookup = dict(zip(seed_steps.tolist(), seed_values.tolist()))
                rows.append([lookup[s] for s in common_steps])
            values = np.asarray(rows, dtype=float)
            mean = values.mean(axis=0)
            std = values.std(axis=0, ddof=1)
            sem = std / math.sqrt(values.shape[0])
            aggregated[env][method] = {
                "steps": steps,
                "seed_values": values,
                "mean": mean,
                "std": std,
                "sem": sem,
                "ci95": 1.96 * sem,
            }
    return aggregated


def write_csv(aggregated):
    path = OUT / "procgen_rat_vs_ppo_curves_5seed.csv"
    with path.open("w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(
            [
                "environment",
                "method",
                "steps",
                "n_seeds",
                "mean_eprewmean",
                "std_eprewmean",
                "sem_eprewmean",
                "ci95_halfwidth",
            ]
        )
        for env in ENV_ORDER:
            for method in METHOD_DIRS:
                a = aggregated[env][method]
                for i, step in enumerate(a["steps"]):
                    writer.writerow(
                        [
                            ENV_LABEL[env],
                            method,
                            int(step),
                            5,
                            f"{a['mean'][i]:.8g}",
                            f"{a['std'][i]:.8g}",
                            f"{a['sem'][i]:.8g}",
                            f"{a['ci95'][i]:.8g}",
                        ]
                    )
    return path


def write_json(aggregated):
    payload = {
        "meta": {
            "metric": "eprewmean",
            "aggregation": "mean with 95% normal-approximation CI across 5 seeds",
            "horizon": 6_000_000,
            "methods": list(METHOD_DIRS),
        },
        "environments": [],
    }
    for env in ENV_ORDER:
        panel = {"id": env, "label": ENV_LABEL[env], "series": []}
        for method in METHOD_DIRS:
            a = aggregated[env][method]
            # Keep the first and final point and about 48 evenly spaced intermediates.
            indices = np.unique(
                np.linspace(0, len(a["steps"]) - 1, 50).round().astype(int)
            )
            panel["series"].append(
                {
                    "method": method,
                    "steps": [int(a["steps"][i]) for i in indices],
                    "mean": [round(float(a["mean"][i]), 5) for i in indices],
                    "ci95": [round(float(a["ci95"][i]), 5) for i in indices],
                    "final_mean": round(float(a["mean"][-1]), 4),
                }
            )
        payload["environments"].append(panel)
    path = ROOT / "curve_data.json"
    path.write_text(json.dumps(payload, separators=(",", ":")))
    return path


def make_plot(aggregated):
    width, height = 2400, 3000
    image = Image.new("RGB", (width, height), "white")
    draw = ImageDraw.Draw(image, "RGBA")
    font_path = "/System/Library/Fonts/Supplemental/Arial.ttf"
    try:
        title_font = ImageFont.truetype(font_path, 48)
        panel_font = ImageFont.truetype(font_path, 34)
        label_font = ImageFont.truetype(font_path, 25)
        tick_font = ImageFont.truetype(font_path, 22)
    except OSError:
        title_font = panel_font = label_font = tick_font = ImageFont.load_default()

    colors = {"PPO": (47, 109, 179), "Exact RAT": (217, 120, 35)}
    draw.text(
        (width / 2, 38),
        "Procgen PPO vs Exact RAT — 7a0698-derived, not target 2b5affd (5 seeds)",
        fill=(25, 25, 25),
        font=title_font,
        anchor="ma",
    )
    legend_y = 112
    legend_items = [("PPO", width / 2 - 220), ("Exact RAT", width / 2 + 40)]
    for method, x in legend_items:
        draw.line((x, legend_y, x + 70, legend_y), fill=colors[method], width=7)
        draw.text((x + 85, legend_y), method, fill=(35, 35, 35), font=label_font, anchor="lm")

    outer_x, outer_top, gap_x, gap_y = 110, 175, 100, 72
    panel_w = (width - 2 * outer_x - gap_x) / 2
    panel_h = (height - outer_top - 100 - 3 * gap_y) / 4
    for index, env in enumerate(ENV_ORDER):
        row, col = divmod(index, 2)
        x0 = outer_x + col * (panel_w + gap_x)
        y0 = outer_top + row * (panel_h + gap_y)
        plot_left, plot_right = x0 + 115, x0 + panel_w - 35
        plot_top, plot_bottom = y0 + 62, y0 + panel_h - 92

        lowers, uppers = [], []
        for method in METHOD_DIRS:
            a = aggregated[env][method]
            lowers.append(a["mean"] - a["ci95"])
            uppers.append(a["mean"] + a["ci95"])
        y_min = min(float(np.nanmin(v)) for v in lowers)
        y_max = max(float(np.nanmax(v)) for v in uppers)
        if not y_max > y_min:
            y_max = y_min + 1
        pad = 0.06 * (y_max - y_min)
        y_min, y_max = y_min - pad, y_max + pad

        def sx(step):
            return plot_left + (float(step) / 6_000_000.0) * (plot_right - plot_left)

        def sy(value):
            return plot_bottom - ((float(value) - y_min) / (y_max - y_min)) * (plot_bottom - plot_top)

        for frac in np.linspace(0, 1, 5):
            yy = plot_bottom - frac * (plot_bottom - plot_top)
            value = y_min + frac * (y_max - y_min)
            draw.line((plot_left, yy, plot_right, yy), fill=(205, 205, 205, 145), width=1)
            draw.text((plot_left - 12, yy), f"{value:.2g}", fill=(75, 75, 75), font=tick_font, anchor="rm")
        for million in range(0, 7, 2):
            xx = sx(million * 1_000_000)
            draw.line((xx, plot_top, xx, plot_bottom), fill=(220, 220, 220, 110), width=1)
            draw.text((xx, plot_bottom + 14), str(million), fill=(75, 75, 75), font=tick_font, anchor="ma")
        draw.line((plot_left, plot_top, plot_left, plot_bottom), fill=(65, 65, 65), width=2)
        draw.line((plot_left, plot_bottom, plot_right, plot_bottom), fill=(65, 65, 65), width=2)

        for method in METHOD_DIRS:
            a = aggregated[env][method]
            lower = a["mean"] - a["ci95"]
            upper = a["mean"] + a["ci95"]
            top_points = [(sx(s), sy(v)) for s, v in zip(a["steps"], upper)]
            bottom_points = [(sx(s), sy(v)) for s, v in zip(a["steps"][::-1], lower[::-1])]
            draw.polygon(top_points + bottom_points, fill=(*colors[method], 38))
            line_points = [(sx(s), sy(v)) for s, v in zip(a["steps"], a["mean"])]
            draw.line(line_points, fill=colors[method], width=5, joint="curve")
            ex, ey = line_points[-1]
            draw.ellipse((ex - 6, ey - 6, ex + 6, ey + 6), fill=colors[method])

        draw.text((x0 + panel_w / 2, y0 + 10), ENV_LABEL[env], fill=(25, 25, 25), font=panel_font, anchor="ma")
        draw.text((x0 + panel_w / 2, plot_bottom + 54), "Environment steps (millions)", fill=(55, 55, 55), font=label_font, anchor="ma")
    label_layer = Image.new("RGBA", (520, 70), (255, 255, 255, 0))
    label_draw = ImageDraw.Draw(label_layer)
    label_draw.text(
        (260, 35),
        "Episode return (eprewmean)",
        fill=(55, 55, 55),
        font=label_font,
        anchor="mm",
    )
    label_layer = label_layer.rotate(90, expand=True)
    image.paste(label_layer, (5, int((height - label_layer.height) / 2)), label_layer)

    png = OUT / "procgen_rat_vs_ppo_curves_5seed.png"
    pdf = OUT / "procgen_rat_vs_ppo_curves_5seed.pdf"
    image.save(png)
    image.save(pdf, "PDF", resolution=180.0)
    return png, pdf


def write_validation(curves, aggregated):
    path = OUT / "procgen_rat_vs_ppo_curve_validation.md"
    lines = [
        "# Procgen PPO vs Exact RAT curve validation",
        "",
        "> **Version warning:** These curves come from runs derived from desktop HEAD `7a0698e`, "
        "not the requested commit `2b5affd64cbb3c624b4bc1f4767f449df231ffb2`. "
        "They must not be reported as results for the requested commit.",
        "",
        "- Extracted from completed 4090 stdout logs on 2026-07-22.",
        "- Metric: `eprewmean` (trainer's rolling episodic-return statistic).",
        "- Aggregation: arithmetic mean with 95% normal-approximation CI across 5 seeds.",
        "- No additional temporal smoothing.",
        "- All 80 seed logs have return code 0.",
        "",
        "| Environment | PPO seeds | RAT seeds | PPO points/seed | RAT points/seed | PPO final mean | RAT final mean |",
        "|---|---:|---:|---:|---:|---:|---:|",
    ]
    for env in ENV_ORDER:
        ppo_points = [len(curves["PPO"][env][s][0]) for s in range(5)]
        rat_points = [len(curves["Exact RAT"][env][s][0]) for s in range(5)]
        ppo_final = aggregated[env]["PPO"]["mean"][-1]
        rat_final = aggregated[env]["Exact RAT"]["mean"][-1]
        lines.append(
            f"| {ENV_LABEL[env]} | 5 | 5 | {min(ppo_points)}-{max(ppo_points)} | "
            f"{min(rat_points)}-{max(rat_points)} | {ppo_final:.3g} | {rat_final:.3g} |"
        )
    path.write_text("\n".join(lines) + "\n")
    return path


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    curves = collect()
    aggregated = aggregate(curves)
    products = [
        write_csv(aggregated),
        write_json(aggregated),
        *make_plot(aggregated),
        write_validation(curves, aggregated),
    ]
    for product in products:
        print(product)


if __name__ == "__main__":
    main()
