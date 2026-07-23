"""Build the paper's gate-1 program figures from committed artifacts.

Every number in these figures is read at build time from committed
``runs/`` artifacts and ``gates.yaml`` — nothing is hand-entered. The
SVGs are deterministic functions of the committed evidence, so a
rebuild on any checkout reproduces them byte-for-byte.

Outputs (paper/figures/):
  autocorr_ladder.svg  — the 12 registered runs against the locked
                         long-horizon autocorrelation bands
  gate_scorecard.svg   — per-seed geometry/battery conjunction for
                         every registered run, with gate verdicts
  c2st_noise.svg       — the 20-seed pairs-C2ST extension of run 12
                         against the matched real-vs-real floor

Usage: python scripts/build_paper_figures.py
"""

from __future__ import annotations

import json
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "paper" / "figures"

# Registration order (issue #42); labels match the paper's ladder table.
RUNS = [
    ("gate1_qrf_baseline_v1", "chained weighted QRF (baseline)"),
    ("gate1_qrf_latent_perm_v1", "+ person effect as feature"),
    ("gate1_qrf_latent_perm_v2", "+ persistence-aware decomposition"),
    ("gate1_qrf_structural_v1", "structural three-component assembly"),
    ("gate1_splice_v1", "donor splicing (single-donor)"),
    ("gate1_splice_v2", "donor splicing (segments)"),
    ("gate1_rank_v1", "Gaussian-copula rank dynamics"),
    ("gate1_rank_kernel_v1", "empirical rank kernel"),
    ("gate1_rank_knn_v1", "k-NN rank bootstrap"),
    ("gate1_rank_knn_v2", "permanent-rank matching"),
    ("gate1_rank_knn_v3", "calibrated blend + Q0 regime"),
    ("gate1_rank_knn_v4", "inner-validated composition"),
]
# Run 13 (candidate 11) re-registered the run-12 spec under ratified
# amendment 2 and reproduced it bit-exactly, so its ladder curve
# coincides with run 12's; it appears in the scorecard only.
SCORECARD_RUNS = RUNS + [
    ("gate1_rank_knn_v5", "inner-validated composition (re-registered)"),
]
HORIZONS = (2, 4, 10)

# Palette (validated, light surface): categorical blue/orange, neutral
# greys for context, status green/red for verdict glyphs.
BLUE = "#2a78d6"
ORANGE = "#eb6834"
GOOD = "#0ca30c"
BAD = "#e34948"
INK = "#0b0b0b"
MUTED = "#52514e"
CONTEXT = "#b2b0a9"
GRID = "#e8e7e3"
FONT = "font-family='system-ui, -apple-system, sans-serif'"


def load_artifacts(runs: list | None = None) -> list[dict]:
    return [
        json.loads((ROOT / "runs" / f"{run}.json").read_text())
        for run, _ in (runs or RUNS)
    ]


def load_bands() -> dict[int, tuple[float, float]]:
    """Locked autocorrelation bands: committed reference ± tolerance."""
    gates = yaml.safe_load((ROOT / "gates.yaml").read_text())
    battery = gates["gates"]["gate_1"]["thresholds"]["battery"]
    reference = json.loads(
        (ROOT / "runs" / "noise_floor_psid_family_9822.json").read_text()
    )["battery_reference"]
    return {
        h: (
            reference[f"autocorr_log_{h}yr"],
            battery[f"autocorr_log_{h}yr_tolerance"],
        )
        for h in HORIZONS
    }


def esc(text: str) -> str:
    return text.replace("&", "&amp;").replace("<", "&lt;")


def text(
    x: float,
    y: float,
    s: str,
    size: int = 12,
    fill: str = INK,
    anchor: str = "start",
    weight: str = "normal",
) -> str:
    return (
        f"<text x='{x:.1f}' y='{y:.1f}' {FONT} font-size='{size}' "
        f"fill='{fill}' text-anchor='{anchor}' "
        f"font-weight='{weight}'>{esc(s)}</text>"
    )


def svg(width: int, height: int, body: list[str]) -> str:
    return (
        f"<svg viewBox='0 0 {width} {height}' width='{width}' "
        f"height='{height}' xmlns='http://www.w3.org/2000/svg' "
        f"style='max-width:100%;height:auto;background:#ffffff'>\n"
        + "\n".join(body)
        + "\n</svg>\n"
    )


# --------------------------------------------------------------------------
# Figure 1: the autocorrelation ladder
# --------------------------------------------------------------------------
def build_ladder(artifacts: list[dict]) -> str:
    bands = load_bands()
    W, H = 760, 430
    L, R, T, B = 64, 258, 28, 46
    xs = {h: L + i * (W - L - R) / 2 for i, h in enumerate(HORIZONS)}
    y_lo, y_hi = 0.28, 0.84

    def ys(v: float) -> float:
        return T + (y_hi - v) / (y_hi - y_lo) * (H - T - B)

    body: list[str] = []
    # y grid + ticks
    tick = 0.3
    while tick <= 0.81:
        y = ys(tick)
        body.append(
            f"<line x1='{L - 6}' y1='{y:.1f}' x2='{W - R + 40}' "
            f"y2='{y:.1f}' stroke='{GRID}' stroke-width='1'/>"
        )
        body.append(text(L - 12, y + 4, f"{tick:.1f}", 11, MUTED, "end"))
        tick = round(tick + 0.1, 1)
    body.append(
        f"<text x='16' y='{(T + H - B) / 2:.0f}' {FONT} font-size='12' "
        f"fill='{MUTED}' text-anchor='middle' transform='rotate(-90 16 "
        f"{(T + H - B) / 2:.0f})'>autocorrelation of log earnings</text>"
    )

    # Locked bands (reference ± tolerance) + reference tick
    for h in HORIZONS:
        ref, tol = bands[h]
        x = xs[h]
        body.append(
            f"<rect x='{x - 30:.1f}' y='{ys(ref + tol):.1f}' width='60' "
            f"height='{ys(ref - tol) - ys(ref + tol):.1f}' "
            f"fill='{MUTED}' opacity='0.10' rx='3'/>"
        )
        body.append(
            f"<line x1='{x - 30:.1f}' y1='{ys(ref):.1f}' "
            f"x2='{x + 30:.1f}' y2='{ys(ref):.1f}' stroke='{MUTED}' "
            f"stroke-width='1.5' stroke-dasharray='4 3'/>"
        )
        body.append(text(x, H - B + 22, f"{h}-year lag", 12, MUTED, "middle"))
    band_label_y = ys(bands[10][0]) - 34
    body.append(text(xs[10] + 38, band_label_y, "locked band", 11, MUTED))
    body.append(
        text(
            xs[10] + 38,
            band_label_y + 14,
            "(PSID reference ± tolerance)",
            11,
            MUTED,
        )
    )

    # Run lines: context greys first, then the two highlighted runs.
    series = []
    for (run, label), art in zip(RUNS, artifacts, strict=True):
        per = art["per_seed"]
        vals = {
            h: sum(s["battery_values"][f"autocorr_log_{h}yr"] for s in per)
            / len(per)
            for h in HORIZONS
        }
        series.append((run, label, vals))

    def polyline(vals: dict, color: str, width: float, opacity: float):
        pts = " ".join(f"{xs[h]:.1f},{ys(vals[h]):.1f}" for h in HORIZONS)
        return (
            f"<polyline points='{pts}' fill='none' stroke='{color}' "
            f"stroke-width='{width}' opacity='{opacity}' "
            f"stroke-linejoin='round'/>"
        )

    highlights = {
        "gate1_qrf_baseline_v1": (ORANGE, "run 1: chained QRF baseline"),
        "gate1_rank_knn_v4": (BLUE, "run 12: inner-validated composition"),
    }
    for run, _label, vals in series:
        if run not in highlights:
            body.append(polyline(vals, CONTEXT, 1.5, 0.75))
    label_slots: list[tuple[float, str, str]] = []
    for run, _label, vals in series:
        if run in highlights:
            color, tag = highlights[run]
            body.append(polyline(vals, color, 2.5, 1.0))
            for h in HORIZONS:
                body.append(
                    f"<circle cx='{xs[h]:.1f}' cy='{ys(vals[h]):.1f}' "
                    f"r='4.5' fill='{color}'/>"
                )
            label_slots.append((ys(vals[10]), tag, color))
        elif run == "gate1_rank_knn_v2":
            label_slots.append(
                (ys(vals[10]), "run 10: permanent-rank matching", CONTEXT)
            )

    # Direct labels at the right edge, de-overlapped.
    label_slots.sort()
    placed: list[float] = []
    for y, tag, color in label_slots:
        while any(abs(y - p) < 15 for p in placed):
            y += 15
        placed.append(y)
        fill = MUTED if color == CONTEXT else color
        weight = "normal" if color == CONTEXT else "600"
        body.append(text(xs[10] + 38, y + 4, tag, 12, fill, "start", weight))

    body.append(
        text(L, T - 8, "other registered runs shown in grey", 11, MUTED)
    )
    return svg(W, H, body)


# --------------------------------------------------------------------------
# Figure 2: the 20-seed pairs-C2ST noise measurement
# --------------------------------------------------------------------------
def build_c2st_noise() -> str:
    art = json.loads((ROOT / "runs" / "c10_diagnostics_v1.json").read_text())
    ext = art["diagnostic_1_seed_extension"]
    per = ext["per_seed"]
    threshold = ext["pairs_c2st_threshold"]
    cand = [(r["seed"], r["candidate_pairs_c2st"], r["source"]) for r in per]
    floor = [(r["seed"], r["floor_pairs_c2st"], r["source"]) for r in per]
    mean20 = ext["candidate_c2st_distribution"]["mean"]
    locked5 = [v for s, v, src in cand if src.startswith("committed")]
    mean_locked5 = sum(locked5) / len(locked5)
    floor_mean = ext["floor_c2st_distribution"]["mean"]

    W, H = 760, 300
    L, R = 64, 40
    x_lo = min(v for _, v, _ in cand + floor) - 0.004
    x_hi = max(v for _, v, _ in cand + floor) + 0.004
    x_hi = max(x_hi, threshold + 0.004)

    def xp(v: float) -> float:
        return L + (v - x_lo) / (x_hi - x_lo) * (W - L - R)

    y_cand, y_floor, y_axis = 108, 196, 248
    body: list[str] = []

    # Axis + ticks
    body.append(
        f"<line x1='{L}' y1='{y_axis}' x2='{W - R}' y2='{y_axis}' "
        f"stroke='{GRID}' stroke-width='1.5'/>"
    )
    t = round(x_lo * 100) / 100
    while t <= x_hi:
        if t >= x_lo:
            body.append(
                f"<line x1='{xp(t):.1f}' y1='{y_axis}' x2='{xp(t):.1f}' "
                f"y2='{y_axis + 5}' stroke='{MUTED}' stroke-width='1'/>"
            )
            body.append(
                text(xp(t), y_axis + 20, f"{t:.2f}", 11, MUTED, "middle")
            )
        t = round(t + 0.01, 2)
    body.append(
        text(
            (L + W - R) / 2,
            y_axis + 40,
            "pairs-view C2ST AUC (0.5 = indistinguishable from real)",
            12,
            MUTED,
            "middle",
        )
    )

    # Locked per-seed line
    body.append(
        f"<line x1='{xp(threshold):.1f}' y1='40' x2='{xp(threshold):.1f}' "
        f"y2='{y_axis}' stroke='{BAD}' stroke-width='1.5' "
        f"stroke-dasharray='5 4'/>"
    )
    body.append(
        text(
            xp(threshold),
            30,
            f"locked per-seed line ({threshold})",
            11,
            BAD,
            "middle",
            "600",
        )
    )

    def strip(rows, y, color, label):
        body.append(text(L, y - 26, label, 12, INK, "start", "600"))
        for _seed, v, src in rows:
            x = xp(v)
            if src.startswith("committed"):
                body.append(
                    f"<circle cx='{x:.1f}' cy='{y}' r='7.5' fill='none' "
                    f"stroke='{color}' stroke-width='1.5'/>"
                )
            body.append(
                f"<circle cx='{x:.1f}' cy='{y}' r='4.5' fill='{color}' "
                f"opacity='0.85'/>"
            )

    strip(cand, y_cand, BLUE, "candidate 10 vs real (20 seeds)")
    strip(floor, y_floor, MUTED, "real vs real, matched floor (20 seeds)")

    # Label the two clipping locked seeds (the global max and 2nd max).
    clipping = sorted(
        (r for r in cand if r[1] > threshold), key=lambda r: r[1]
    )
    for i, (seed, v, _) in enumerate(clipping):
        # Alternate anchors so adjacent labels never collide.
        anchor = "end" if i % 2 == 0 else "start"
        dx = -2 if anchor == "end" else 2
        body.append(
            text(xp(v) + dx, y_cand - 14, f"seed {seed}", 10, BLUE, anchor)
        )

    # Mean ticks: both means, per the amendment-2 referee record.
    for v, label, color, y0 in (
        (mean20, f"20-seed mean {mean20:.4f}", BLUE, y_cand + 16),
        (mean_locked5, f"locked-5 mean {mean_locked5:.4f}", BLUE, y_cand + 34),
        (floor_mean, f"floor mean {floor_mean:.4f}", MUTED, y_floor + 16),
    ):
        body.append(
            f"<line x1='{xp(v):.1f}' y1='{y0 - 12}' x2='{xp(v):.1f}' "
            f"y2='{y0 + 2}' stroke='{color}' stroke-width='2.5'/>"
        )
        anchor = "end" if v < threshold else "start"
        dx = -6 if anchor == "end" else 6
        body.append(text(xp(v) + dx, y0, label, 11, color, anchor))

    body.append(
        text(
            L,
            60,
            "ringed = the five locked gate seeds",
            11,
            MUTED,
        )
    )
    return svg(W, H, body)


# --------------------------------------------------------------------------
# Figure 3: the gate scorecard
# --------------------------------------------------------------------------
def build_scorecard(artifacts: list[dict]) -> str:
    n = len(artifacts)
    W = 760
    row_h = 27
    top = 64
    H = top + n * row_h + 72
    x_label, x_geo, x_bat, x_q0, x_verdict = 20, 388, 508, 618, 686
    body: list[str] = []

    body.append(text(x_label, 24, "registered run", 12, MUTED))
    body.append(text(x_geo, 24, "geometry", 12, MUTED, "middle"))
    body.append(text(x_bat, 24, "battery", 12, MUTED, "middle"))
    body.append(text(x_geo, 40, "seeds passed (of 5)", 10, MUTED, "middle"))
    body.append(text(x_bat, 40, "seeds passed (of 5)", 10, MUTED, "middle"))
    body.append(text(x_q0, 24, "pooled Q0", 12, MUTED, "middle"))
    body.append(text(x_verdict, 24, "gate 1", 12, MUTED, "middle"))

    for i, ((_run, label), art) in enumerate(
        zip(SCORECARD_RUNS, artifacts, strict=True)
    ):
        y = top + i * row_h
        if i % 2 == 0:
            body.append(
                f"<rect x='10' y='{y - 17}' width='{W - 20}' height='{row_h}'"
                f" fill='{MUTED}' opacity='0.05' rx='3'/>"
            )
        body.append(text(x_label, y, f"{i + 1}.", 12, MUTED))
        body.append(text(x_label + 24, y, label, 12, INK))

        for cx, key in ((x_geo, "geometry_pass"), (x_bat, "battery_pass")):
            seeds = sorted(art["per_seed"], key=lambda s: s["seed"])
            for j, s in enumerate(seeds):
                x = cx + (j - 2) * 15
                if s[key]:
                    body.append(
                        f"<circle cx='{x}' cy='{y - 4}' r='5' "
                        f"fill='{BLUE}'/>"
                    )
                else:
                    body.append(
                        f"<circle cx='{x}' cy='{y - 4}' r='5' fill='none' "
                        f"stroke='{CONTEXT}' stroke-width='1.5'/>"
                    )

        q0_pass = art["verdict"].get("pooled_q0_pass")
        if q0_pass is None:
            body.append(text(x_q0, y, "not scored", 10, MUTED, "middle"))
        else:
            glyph, color = ("✓", GOOD) if q0_pass else ("✕", BAD)
            body.append(text(x_q0, y, glyph, 13, color, "middle", "700"))

        gate_pass = art["verdict"]["gate_1_pass"]
        glyph, word, color = (
            ("✓", "pass", GOOD) if gate_pass else ("✕", "fail", BAD)
        )
        body.append(text(x_verdict - 14, y, glyph, 13, color, "middle", "700"))
        body.append(text(x_verdict - 2, y, word, 12, color))

    y_note = top + n * row_h + 6
    body.append(
        f"<circle cx='{x_label + 4}' cy='{y_note - 4}' r='5' "
        f"fill='{BLUE}'/>"
    )
    body.append(text(x_label + 14, y_note, "seed passed", 11, MUTED))
    body.append(
        f"<circle cx='{x_label + 104}' cy='{y_note - 4}' r='5' fill='none' "
        f"stroke='{CONTEXT}' stroke-width='1.5'/>"
    )
    body.append(text(x_label + 114, y_note, "seed failed", 11, MUTED))
    body.append(
        text(
            x_label,
            y_note + 20,
            "gate 1 requires ≥4/5 seeds on geometry AND ≥4/5 on battery;",
            11,
            MUTED,
        )
    )
    body.append(
        text(
            x_label,
            y_note + 34,
            "runs 11–13 were also scored on the amended benefit-space "
            "block and pooled Q0; run 13's pairs-view classifier is "
            "gated by the",
            11,
            MUTED,
        )
    )
    body.append(
        text(
            x_label,
            y_note + 48,
            "amendment-2 rule (mean over 20 pre-registered seeds ≤ "
            "0.53 with a per-seed cap of 0.554) rather than per-seed "
            "at 0.53.",
            11,
            MUTED,
        )
    )
    return svg(W, H, body)




# --------------------------------------------------------------------------
# Figure 4: the M6 candidate scorecard (candidates 1 and 2, per seed)
# --------------------------------------------------------------------------
def build_m6_scorecard() -> str:
    d1 = json.loads((ROOT / "runs/gate_m6_candidate1_v1.json").read_text())
    d2 = json.loads((ROOT / "runs/gate_m6_candidate2_v1.json").read_text())
    tol = {c["cell"]: c["tolerance"] for c in d1["gate"]["cells"]}

    def norm(per_seed: list[dict]) -> dict[str, list[float]]:
        out: dict[str, list[float]] = {c: [] for c in tol}
        for s in per_seed:
            for c in tol:
                out[c].append(s["gated_cells"][c]["score"] / tol[c])
        return out

    s1 = norm(d1["family_a"]["per_seed"])
    s2 = norm(
        d2["candidate2_acceptance"]["gate_contract_result"]["per_seed"]
    )
    order = sorted(tol, key=lambda c: sum(s2[c]) / len(s2[c]))
    x_max = max(max(v) for v in s1.values()) * 1.05

    W, H = 760, 470
    L, R, T, B = 214, 24, 54, 56
    row_h = (H - T - B) / len(order)

    def xs(v: float) -> float:
        return L + v / x_max * (W - L - R)

    body: list[str] = []
    body.append(
        text(L, 22, "Two registered candidates against the locked "
             "gate_m6 holdout", 14, INK, weight="bold")
    )
    body.append(
        text(L, 40, "fit \u2264 2014, scored on held-out 2015\u20132022; "
             "holdout error / locked tolerance, five seeds each", 11, MUTED)
    )
    for gx in (0.0, 1.0, 2.0, 3.0, 4.0):
        if gx > x_max:
            continue
        body.append(
            f"<line x1='{xs(gx):.1f}' y1='{T}' x2='{xs(gx):.1f}' "
            f"y2='{H - B}' stroke='{GRID}' stroke-width='1'/>"
        )
        body.append(
            text(xs(gx), H - B + 16, f"{gx:.0f}", 10, MUTED, "middle")
        )
    body.append(
        f"<line x1='{xs(1.0):.1f}' y1='{T - 6}' x2='{xs(1.0):.1f}' "
        f"y2='{H - B}' stroke='{INK}' stroke-width='1.6'/>"
    )
    body.append(
        text(xs(1.0) + 5, T + 6, "locked tolerance", 10, INK)
    )
    for i, c in enumerate(order):
        cy = T + (i + 0.5) * row_h
        body.append(text(L - 8, cy + 4, c, 11, INK, "end"))
        for v in s1[c]:
            body.append(
                f"<circle cx='{xs(v):.1f}' cy='{cy - 4:.1f}' r='4.6' "
                f"fill='{ORANGE}' fill-opacity='0.9' stroke='#ffffff' "
                f"stroke-width='1.2'/>"
            )
        for v in s2[c]:
            body.append(
                f"<circle cx='{xs(v):.1f}' cy='{cy + 5:.1f}' r='4.6' "
                f"fill='{BLUE}' fill-opacity='0.92' stroke='#ffffff' "
                f"stroke-width='1.2'/>"
            )
    ly = H - B + 34
    body.append(
        f"<circle cx='{L + 6}' cy='{ly - 4}' r='4.6' fill='{ORANGE}'/>"
    )
    body.append(
        text(L + 16, ly, "candidate 1 (FAIL, 6 of 11 cells)", 11, INK)
    )
    body.append(
        f"<circle cx='{L + 256}' cy='{ly - 4}' r='4.6' fill='{BLUE}'/>"
    )
    body.append(
        text(L + 266, ly, "candidate 2 (FAIL, 3 of 5 seeds)", 11, INK)
    )
    return svg(W, H, body)


# --------------------------------------------------------------------------
# Figure 5: the train-only persistence-mobility frontier (q ladder)
# --------------------------------------------------------------------------
def build_m6_q_frontier() -> str:
    led = json.loads(
        (ROOT / "docs/analysis/m6_qstar_train_only_selection_results.json")
        .read_text()
    )
    qs: list[float] = []
    auto: list[float] = []
    mob: list[float] = []
    for q in sorted(led["rungs"], key=float):
        a = m = 0.0
        for b in ("2006", "2008", "2010"):
            oc = led["rungs"][q]["boundaries"][b]["aggregates"]["all_20"][
                "objective_contributions"
            ]
            a += oc["earn_autocorr_lag2"]
            m += oc["earn_mob_h1_diag"]
        qs.append(float(q))
        auto.append(a)
        mob.append(m)
    y_max = max(max(auto), max(mob)) * 1.06

    W, H = 760, 430
    L, R, T, B = 64, 172, 54, 46

    def xf(q: float) -> float:
        return L + q * (W - L - R)

    def yf(v: float) -> float:
        return T + (1 - v / y_max) * (H - T - B)

    def path(vals: list[float]) -> str:
        return " ".join(
            f"{'M' if i == 0 else 'L'}{xf(q):.1f},{yf(v):.1f}"
            for i, (q, v) in enumerate(zip(qs, vals))
        )

    body: list[str] = []
    body.append(
        text(L, 22, "The train-only persistence\u2013mobility frontier "
             "behind candidate 3", 14, INK, weight="bold")
    )
    body.append(
        text(L, 40, "standardized objective contribution by refresh "
             "share q, summed over the three pseudo-boundaries", 11, MUTED)
    )
    for gy in range(0, int(y_max) + 1, 100):
        body.append(
            f"<line x1='{L}' y1='{yf(gy):.1f}' x2='{W - R}' "
            f"y2='{yf(gy):.1f}' stroke='{GRID}' stroke-width='1'/>"
        )
        body.append(text(L - 8, yf(gy) + 4, str(gy), 10, MUTED, "end"))
    for gx in (0.0, 0.2, 0.4, 0.6, 0.8, 1.0):
        body.append(
            text(xf(gx), H - B + 16, f"{gx:.1f}", 10, MUTED, "middle")
        )
    body.append(
        f"<line x1='{xf(0.55):.1f}' y1='{T}' x2='{xf(0.55):.1f}' "
        f"y2='{H - B}' stroke='{INK}' stroke-width='1.2' "
        f"stroke-dasharray='5 4'/>"
    )
    body.append(text(xf(0.55) + 5, T + 14, "selected q* = 0.55", 10, INK))
    body.append(
        f"<path d='{path(auto)}' fill='none' stroke='{BLUE}' "
        f"stroke-width='2.4'/>"
    )
    body.append(
        f"<path d='{path(mob)}' fill='none' stroke='{ORANGE}' "
        f"stroke-width='2.4'/>"
    )
    body.append(
        text(xf(1.0) + 8, yf(auto[-1]) + 4, "lag-2 autocorrelation", 11,
             BLUE)
    )
    body.append(
        text(xf(1.0) + 8, yf(mob[-1]) + 4, "rank mobility", 11, ORANGE)
    )
    body.append(
        text(W - R, H - B + 32, "refresh share q (train-only ladder)", 10,
             MUTED, "end")
    )
    return svg(W, H, body)


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "autocorr_ladder.svg").write_text(build_ladder(load_artifacts()))
    (OUT / "c2st_noise.svg").write_text(build_c2st_noise())
    (OUT / "m6_candidate_scorecard.svg").write_text(build_m6_scorecard())
    (OUT / "m6_q_frontier.svg").write_text(build_m6_q_frontier())
    (OUT / "gate_scorecard.svg").write_text(
        build_scorecard(load_artifacts(SCORECARD_RUNS))
    )
    for name in ("autocorr_ladder", "c2st_noise", "gate_scorecard"):
        print(f"wrote paper/figures/{name}.svg")


if __name__ == "__main__":
    main()
