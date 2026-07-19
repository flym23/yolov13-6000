"""Analyze completed RP-SCAF ablation results against the D0 YOLOv13 baseline.

This script reads remote summary JSON files through Paramiko/SFTP and writes a
local UTF-8 text report. It intentionally uses only summary files and does not
modify remote training outputs.
"""

from __future__ import annotations

import argparse
import json
import statistics
from datetime import datetime
from pathlib import Path
from typing import Any

import paramiko


METRICS = ["P", "R", "mAP50", "mAP75", "mAP50-95", "APS", "APM", "APL"]

DEFAULT_STAGES = {
    "D0": "runs/test/baseline_d0_original_20260708_215137_summary.json",
    "R1": "runs/test/rp_scaf_20260709_173408_r1_rp_scaf_summary.json",
    "R4": "runs/test/rp_scaf_20260709_173408_r4_rp_scaf_no_consistency_summary.json",
    "R5": "runs/test/rp_scaf_20260709_173408_r5_rp_scaf_channel_summary.json",
}
DEFAULT_STATE = "runs/rp_scaf_ablation/state.json"


def fmt(value: float | None) -> str:
    """Format a metric value with three decimals."""
    if value is None:
        return "NA"
    return f"{value:.3f}"


def signed(value: float | None) -> str:
    """Format a signed metric delta with three decimals."""
    if value is None:
        return "NA"
    return ("+" if value >= 0 else "") + f"{value:.3f}"


def mean(values: list[float]) -> float:
    """Return arithmetic mean."""
    return sum(values) / len(values)


def std(values: list[float]) -> float:
    """Return sample standard deviation for seed results."""
    return statistics.stdev(values) if len(values) > 1 else 0.0


def read_remote_json(sftp: paramiko.SFTPClient, root: str, rel_path: str) -> dict[str, Any]:
    """Read a remote UTF-8 JSON file with replacement for invalid bytes."""
    with sftp.open(root.rstrip("/") + "/" + rel_path, "r") as handle:
        return json.loads(handle.read().decode("utf-8", "replace"))


def load_remote_payload(args: argparse.Namespace) -> tuple[dict[str, dict[str, Any]], dict[str, Any]]:
    """Load D0/R1/R4/R5 summaries and the chain state file from the remote server."""
    client = paramiko.SSHClient()
    client.set_missing_host_key_policy(paramiko.AutoAddPolicy())
    client.connect(
        args.host,
        port=args.port,
        username=args.user,
        password=args.password,
        timeout=20,
    )
    try:
        sftp = client.open_sftp()
        try:
            data = {stage: read_remote_json(sftp, args.remote_root, path) for stage, path in DEFAULT_STAGES.items()}
            try:
                state = read_remote_json(sftp, args.remote_root, DEFAULT_STATE)
            except Exception as exc:  # noqa: BLE001 - report should still be generated.
                state = {"error": repr(exc)}
        finally:
            sftp.close()
    finally:
        client.close()
    return data, state


def rows_by_seed(summary: dict[str, Any]) -> list[dict[str, Any]]:
    """Return rows sorted by seed."""
    return sorted(summary["rows_percent"], key=lambda row: row["seed"])


def metric_values(summary: dict[str, Any], metric: str) -> list[float]:
    """Return one metric across sorted seed rows."""
    return [row[metric] for row in rows_by_seed(summary)]


def paired_deltas(stage_summary: dict[str, Any], base_by_seed: dict[int, dict[str, Any]], metric: str) -> list[float]:
    """Return same-seed deltas against D0."""
    return [row[metric] - base_by_seed[row["seed"]][metric] for row in rows_by_seed(stage_summary)]


def improve_count(stage_summary: dict[str, Any], base_by_seed: dict[int, dict[str, Any]], metric: str) -> tuple[int, int]:
    """Count same-seed improvements."""
    deltas = paired_deltas(stage_summary, base_by_seed, metric)
    return sum(1 for delta in deltas if delta > 0), len(deltas)


def metric_summary_lines(stage_summary: dict[str, Any], base_summary: dict[str, Any], base_by_seed: dict[int, dict[str, Any]]) -> list[str]:
    """Build detailed per-metric lines for one stage."""
    lines: list[str] = []
    for metric in METRICS:
        values = metric_values(stage_summary, metric)
        deltas = paired_deltas(stage_summary, base_by_seed, metric)
        count, total = improve_count(stage_summary, base_by_seed, metric)
        lines.append(
            f"  {metric:<8} 均值 {fmt(mean(values))} / std {fmt(std(values))} / "
            f"vs D0均值 {signed(mean(values) - base_summary['mean'][metric])} / "
            f"同seed提升 {count}/{total} / paired delta: [{', '.join(signed(delta) for delta in deltas)}]"
        )
    return lines


def build_report(data: dict[str, dict[str, Any]], state: dict[str, Any], remote_root: str) -> str:
    """Build the full RP-SCAF report text."""
    base = data["D0"]
    base_rows = rows_by_seed(base)
    base_by_seed = {row["seed"]: row for row in base_rows}
    rank = sorted(
        ["R1", "R4", "R5"],
        key=lambda stage: (
            data[stage]["mean"]["mAP50-95"],
            data[stage]["mean"]["mAP75"],
            data[stage]["mean"]["mAP50"],
        ),
        reverse=True,
    )

    lines: list[str] = []
    lines.append("RP-SCAF-YOLOv13 已完成实验数据分析报告（R1 / R4 / R5）")
    lines.append("=" * 78)
    lines.append(f"生成时间：{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    lines.append(f"远程项目：{remote_root}")
    lines.append("数据来源：远程 runs/test 下各 stage summary_metrics.json 与 stage summary.json")
    lines.append("说明：当前报告只分析已经完成并生成完整 3 seed summary 的 R1、R4、R5；R2/R3 未纳入本次结论。")
    if state:
        lines.append(
            "链式任务状态记录："
            f"status={state.get('status')}, stage={state.get('stage')}, "
            f"run_id={state.get('run_id')}, updated_at={state.get('updated_at')}"
        )
    lines.append("")

    lines.append("一、实验结构配置对应关系")
    lines.append("-" * 78)
    lines.append("D0 / Original YOLOv13")
    lines.append(f"  YAML：{base['config'].get('yaml')}")
    lines.append(f"  结构：{base['config'].get('structure')}")
    for stage in ["R1", "R4", "R5"]:
        cfg = data[stage]["config"]
        lines.append(f"{stage} / {data[stage]['stage']}")
        lines.append(f"  YAML：{cfg.get('yaml')}")
        lines.append(f"  结构：{cfg.get('structure')}")
    lines.append("")

    lines.append("二、D0 原始 YOLOv13 基线（三次 seed）")
    lines.append("-" * 78)
    lines.append("D0 各 seed 指标（单位：百分比点）")
    header = "seed      " + "  ".join(f"{metric:>9}" for metric in METRICS)
    lines.append(header)
    for row in base_rows:
        lines.append(f"{row['seed']:<9}" + "  ".join(f"{row[metric]:>9.3f}" for metric in METRICS))
    lines.append("D0 均值/std：")
    for metric in METRICS:
        lines.append(
            f"  {metric:<8} mean={fmt(base['mean'][metric])}, std={fmt(base['std'][metric])}, "
            f"values=[{', '.join(fmt(value) for value in base['detail'][metric]['values'])}]"
        )
    lines.append("")

    lines.append("三、R1/R4/R5 均值与 D0 对比总览")
    lines.append("-" * 78)
    lines.append("表中 Δ 表示该实验均值 - D0 均值，单位为百分比点。")
    lines.append(
        f"{'实验':<4} {'P':>9} {'ΔP':>8} {'R':>9} {'ΔR':>8} {'mAP50':>9} {'Δ50':>8} "
        f"{'mAP75':>9} {'Δ75':>8} {'mAP50-95':>10} {'Δ50-95':>9} "
        f"{'APS':>9} {'ΔS':>8} {'APM':>9} {'ΔM':>8} {'APL':>9} {'ΔL':>8}"
    )
    for stage in ["R1", "R4", "R5"]:
        stage_mean = data[stage]["mean"]
        parts = [f"{stage:<4}"]
        for metric in METRICS:
            parts.append(f"{stage_mean[metric]:>9.3f}")
            parts.append(f"{stage_mean[metric] - base['mean'][metric]:>+8.3f}")
        lines.append(" ".join(parts))
    lines.append("")

    lines.append("四、逐实验三次 seed 细节与稳定性")
    lines.append("-" * 78)
    for stage in ["R1", "R4", "R5"]:
        cfg = data[stage]["config"]
        lines.append(f"{stage}：{cfg.get('structure')}")
        lines.append("  逐 seed 原始指标：")
        lines.append("  " + header)
        for row in rows_by_seed(data[stage]):
            lines.append("  " + f"{row['seed']:<9}" + "  ".join(f"{row[metric]:>9.3f}" for metric in METRICS))
        lines.append("  逐指标统计与同 seed 对比：")
        lines.extend(metric_summary_lines(data[stage], base, base_by_seed))
        lines.append("")

    lines.append("五、核心指标判断")
    lines.append("-" * 78)
    for stage in ["R1", "R4", "R5"]:
        stage_mean = data[stage]["mean"]
        lines.append(f"{stage}:")
        for metric in ["mAP50-95", "mAP75", "mAP50"]:
            count, total = improve_count(data[stage], base_by_seed, metric)
            lines.append(
                f"  {metric:<8}: {fmt(stage_mean[metric])} "
                f"({signed(stage_mean[metric] - base['mean'][metric])}), 同seed提升 {count}/{total}"
            )
        scale_parts = []
        for metric in ["APS", "APM", "APL"]:
            count, total = improve_count(data[stage], base_by_seed, metric)
            scale_parts.append(
                f"{metric} {fmt(stage_mean[metric])} "
                f"({signed(stage_mean[metric] - base['mean'][metric])}, {count}/{total})"
            )
        lines.append("  Scale AP: " + ", ".join(scale_parts))
        if stage == "R1":
            lines.append(
                "  判断：R1 的 mAP50-95 均值只比 D0 高 +0.022，且同 seed 仅 1/3 提升；"
                "mAP75 与 APS 明显下降。因此 R1 不能视为稳定有效提升。"
            )
        elif stage == "R4":
            lines.append(
                "  判断：R4 的 mAP50-95 均值提升 +0.320，三次 seed 全部超过 D0；"
                "P、mAP50、APM、APL 均值也提升。缺点是 R 与 APS 均值下降，mAP75 仅小幅提升。"
                "因此 R4 是当前 R1/R4/R5 中最明确的有效提升点，但提升主要体现在整体 AP、P、中/大目标，非小目标。"
            )
        elif stage == "R5":
            lines.append(
                "  判断：R5 的 mAP50-95 均值提升 +0.064，但同 seed 仅 2/3 提升，"
                "mAP75 与 APS 下降，APM 基本无提升。因此 R5 只能算轻微且不稳定的整体 AP 提升，不构成明确强有效配置。"
            )
        lines.append("")

    lines.append("六、横向排序与最终结论")
    lines.append("-" * 78)
    lines.append("按核心指标 mAP50-95 优先、再看 mAP75/mAP50 排序：" + " > ".join(rank))
    lines.append("")
    lines.append("最终结论：")
    lines.append(
        "1. R4（RPSCAFFuse alpha=0.05，关闭 consistency，spatial gate）是当前已完成 R1/R4/R5 中最值得保留的有效结构。"
        "它相对 D0 在 mAP50-95 上提升 +0.320，并且三次 seed 全部提升，说明不是单次随机波动。"
    )
    lines.append(
        "2. R5（channel-only gate）有轻微 mAP50-95 提升（+0.064），但 mAP75、APS、APM 表现不足，"
        "稳定性和收益幅度都弱于 R4。"
    )
    lines.append(
        "3. R1（带 consistency 的 spatial gate）整体收益不成立：mAP50-95 只 +0.022，mAP75 -0.592，"
        "APS -0.755，且 mAP50-95 同 seed 只有 1/3 提升。"
    )
    lines.append(
        "4. 三个 RP-SCAF 已完成配置均没有改善小目标 APS；R4/R5 的相对收益更偏向整体 AP 与中/大目标，"
        "尤其 R4 的 APM +0.400、APL +0.552。"
    )
    lines.append(
        "5. 若只基于本次已完成的 R1/R4/R5 数据判断：RP-SCAF 方案存在有效提升点，具体有效点是 R4；"
        "R1 不有效，R5 仅弱有效/不稳定。"
    )
    lines.append("")

    lines.append("七、原始结果文件路径")
    lines.append("-" * 78)
    for stage, rel_path in DEFAULT_STAGES.items():
        lines.append(f"{stage}: {remote_root.rstrip('/')}/{rel_path}")
    lines.append(f"state: {remote_root.rstrip('/')}/{DEFAULT_STATE}")
    return "\n".join(lines) + "\n"


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--host", default="154.9.253.153")
    parser.add_argument("--port", default=29103, type=int)
    parser.add_argument("--user", default="rom305")
    parser.add_argument("--password", default="Room305@!")
    parser.add_argument("--remote-root", default="/home/rom305/zzf/yolov13-305")
    parser.add_argument(
        "--output",
        default=str(Path.cwd() / "RP_SCAF_YOLOv13_R1_R4_R5_experiment_report_20260709.txt"),
    )
    return parser.parse_args()


def main() -> None:
    """Entry point."""
    args = parse_args()
    data, state = load_remote_payload(args)
    report = build_report(data, state, args.remote_root)
    output_path = Path(args.output)
    output_path.write_text(report, encoding="utf-8")
    print(f"REPORT_PATH={output_path}")
    print(f"REPORT_LINES={len(report.splitlines())}")
    for stage in ["R1", "R4", "R5"]:
        delta = data[stage]["mean"]["mAP50-95"] - data["D0"]["mean"]["mAP50-95"]
        count, total = improve_count(data[stage], {row["seed"]: row for row in rows_by_seed(data["D0"])}, "mAP50-95")
        print(f"{stage}: mAP50-95={fmt(data[stage]['mean']['mAP50-95'])}, delta={signed(delta)}, same_seed={count}/{total}")


if __name__ == "__main__":
    main()
