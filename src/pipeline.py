"""
[TR] Uçtan uca hesaplama: üret -> doğrula -> değerlendir -> kurala göre karar ver.
     Streamlit'ten bağımsızdır:  python -m src.pipeline [--plot PATH] [--explain]
[EN] End-to-end computation: generate -> validate -> evaluate -> rule-based verdict.
     Independent of Streamlit:   python -m src.pipeline [--plot PATH] [--explain]
"""
import argparse
from dataclasses import dataclass

import pandas as pd

from src.generator import OK, FaultyDataGenerator
from src.validator import CHECKS, SensorValidator

DEFAULT_THRESHOLDS = {"temperature": {"min": 0, "max": 150}}
DEFAULT_MAX_FLAGGED_RATIO = 0.01  # [TR] %1'i geçerse UNSAFE  [EN] UNSAFE above 1 %


@dataclass
class Verdict:
    label: str
    flagged_rows: int
    total_rows: int
    max_ratio: float

    @property
    def ratio(self):
        return self.flagged_rows / self.total_rows if self.total_rows else 0.0

    @property
    def reason(self):
        op = ">" if self.label == "UNSAFE" else "<="
        return (f"{self.flagged_rows}/{self.total_rows} rows flagged = {self.ratio:.2%} "
                f"{op} {self.max_ratio:.2%} limit")


def decide(flags, max_ratio=DEFAULT_MAX_FLAGGED_RATIO):
    """[TR] Deterministik karar kuralı: işaretlenen satır oranı > max_ratio ise UNSAFE.
       [EN] Deterministic rule: UNSAFE if the flagged-row ratio exceeds max_ratio."""
    flagged = int(flags.any(axis=1).sum())
    total = len(flags)
    label = "UNSAFE" if total and flagged / total > max_ratio else "SAFE"
    return Verdict(label, flagged, total, max_ratio)


def evaluate(flags, labels):
    """[TR] Her kontrol için: işaretlenen, yakalanan hata, yanlış alarm ve hata tipine göre yakalama.
       [EN] Per check: flagged, caught faults, false alarms and catches per fault type."""
    faulty = labels != OK
    fault_types = [t for t in labels.unique() if t != OK]
    columns = {check: flags[check] for check in CHECKS}
    columns["ANY"] = flags.any(axis=1)

    rows = {}
    for name, mask in columns.items():
        row = {
            "flagged": int(mask.sum()),
            "caught": int((mask & faulty).sum()),
            "false_alarms": int((mask & ~faulty).sum()),
            "missed": int((~mask & faulty).sum()),
        }
        for t in fault_types:
            row[f"{t} (/{int((labels == t).sum())})"] = int((mask & (labels == t)).sum())
        rows[name] = row
    return pd.DataFrame.from_dict(rows, orient="index")


@dataclass
class PipelineResult:
    data: pd.DataFrame
    labels: pd.Series
    flags: pd.DataFrame
    evaluation: pd.DataFrame
    verdict: Verdict

    def summary(self):
        """[TR] LLM'e giden özet (etiketler hariç).  [EN] Summary sent to the LLM (no ground truth)."""
        return {
            "rows": self.verdict.total_rows,
            "flags_per_check": {c: int(self.flags[c].sum()) for c in CHECKS},
            "flagged_rows_any_check": self.verdict.flagged_rows,
            "verdict": self.verdict.label,
            "verdict_rule": self.verdict.reason,
        }


def run_pipeline(n=500, seed=10, thresholds=None, max_ratio=DEFAULT_MAX_FLAGGED_RATIO,
                 z_threshold=3.5, frozen_window=10):
    df, labels = FaultyDataGenerator(seed=seed).generate_data(n)
    validator = SensorValidator(thresholds or DEFAULT_THRESHOLDS,
                                z_threshold=z_threshold, frozen_window=frozen_window)
    flags = validator.validate(df)
    return PipelineResult(df, labels, flags, evaluate(flags, labels), decide(flags, max_ratio))


def format_report(result):
    return "\n".join([
        f"Rows: {len(result.data)}   Faulty rows (ground truth): {int((result.labels != OK).sum())}",
        "",
        result.evaluation.to_string(),
        "",
        f"VERDICT: {result.verdict.label}  ({result.verdict.reason})",
    ])


# [TR] Kontrol renkleri (kategorik slot 1-3)  [EN] Check colours (categorical slots 1-3)
CHECK_COLORS = {"range": "#2a78d6", "robust_z": "#eb6834", "frozen": "#1baf7a"}
CHECK_LABELS = {"range": "Range (0-150)", "robust_z": "Robust Z (|z|>3.5)", "frozen": "Frozen (>=10 equal)"}


def save_plot(result, path):
    """[TR] Sinyal + gerçek hata bölgeleri (üst) ve kontrol başına işaretler (alt).
       [EN] Signal + ground-truth fault spans (top) and per-check flags (bottom)."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    ink, muted, grid = "#0b0b0b", "#52514e", "#e4e3df"
    s = result.data["temperature"]
    x = s.index.to_numpy()
    faulty = (result.labels != OK).to_numpy()

    fig, (ax1, ax2) = plt.subplots(
        2, 1, figsize=(12, 5.6), sharex=True, gridspec_kw={"height_ratios": [3, 1.3], "hspace": 0.08})
    fig.patch.set_facecolor("#fcfcfb")
    for ax in (ax1, ax2):
        ax.set_facecolor("#fcfcfb")
        for side in ("top", "right"):
            ax.spines[side].set_visible(False)
        for side in ("left", "bottom"):
            ax.spines[side].set_color(grid)
        ax.tick_params(colors=muted, labelsize=9)

    # [TR] Gerçek hata bölgeleri  [EN] Ground-truth fault spans
    for fault in [t for t in result.labels.unique() if t != OK]:
        idx = result.labels.index[result.labels == fault]
        ax1.axvspan(idx.min() - 0.5, idx.max() + 0.5, color="#d6d5d0", alpha=0.6, lw=0)
        ax1.annotate(f"true fault: {fault}\nrows {idx.min()}-{idx.max()}",
                     (idx.max() + 4, s.loc[idx].max() if fault != "frozen" else 95),
                     fontsize=9, color=muted, va="center")
    ax1.plot(x, s, color=ink, lw=1.2)
    ax1.set_ylabel("temperature (°C)", color=muted, fontsize=9)
    ax1.grid(axis="y", color=grid, lw=0.8)
    v = result.verdict
    ax1.set_title(f"Verdict: {v.label}  -  {v.reason}", loc="left", fontsize=11, color=ink)

    # [TR] Kontrol başına işaretler; yanlış alarm = x  [EN] Per-check flags; false alarm = x
    for row, check in enumerate(reversed(CHECKS)):
        flagged = result.flags[check].to_numpy()
        hit, fa = flagged & faulty, flagged & ~faulty
        ax2.scatter(x[hit], [row] * hit.sum(), marker="|", s=120, lw=2, color=CHECK_COLORS[check])
        if fa.any():
            ax2.scatter(x[fa], [row] * fa.sum(), marker="x", s=40, lw=1.5, color=ink)
        ev = result.evaluation.loc[check]
        ax2.annotate(f"caught {ev.caught}, false alarms {ev.false_alarms}", (len(x) + 5, row),
                     fontsize=8.5, color=muted, va="center", annotation_clip=False)
    ax2.set_yticks(range(len(CHECKS)))
    ax2.set_yticklabels([CHECK_LABELS[c] for c in reversed(CHECKS)], fontsize=9, color=ink)
    ax2.set_ylim(-0.6, len(CHECKS) - 0.4)
    ax2.set_xlim(-5, len(x) + 5)
    ax2.set_xlabel("reading index  (| = flagged faulty row,  x = false alarm)", color=muted, fontsize=9)

    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def main(argv=None):
    parser = argparse.ArgumentParser(description="Sensor data quality guardrail")
    parser.add_argument("--n", type=int, default=500)
    parser.add_argument("--seed", type=int, default=10)
    parser.add_argument("--max-ratio", type=float, default=DEFAULT_MAX_FLAGGED_RATIO)
    parser.add_argument("--plot", metavar="PATH", help="save a PNG chart (needs matplotlib)")
    parser.add_argument("--explain", action="store_true", help="ask the LLM to explain the verdict")
    args = parser.parse_args(argv)

    result = run_pipeline(n=args.n, seed=args.seed, max_ratio=args.max_ratio)
    print(format_report(result))
    if args.plot:
        save_plot(result, args.plot)
        print(f"\nPlot saved to {args.plot}")
    if args.explain:
        from src.describer import DataQualityDescriber
        print("\n--- LLM explanation ---\n" + DataQualityDescriber().explain(result.summary()))
    return result


if __name__ == "__main__":
    main()
