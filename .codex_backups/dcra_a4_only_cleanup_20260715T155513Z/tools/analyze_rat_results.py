#!/usr/bin/env python3
"""Download RAT-YOLOv13 summaries and generate a three-seed comparison report."""

from __future__ import annotations

import argparse
import base64
import json
import math
import os
import posixpath
import statistics
from datetime import datetime
from pathlib import Path
from typing import Any

import paramiko


METRICS = ["P", "R", "mAP50", "mAP75", "mAP50-95", "APS", "APM", "APL"]
CORE_METRICS = ["mAP50-95", "mAP75", "mAP50", "APS", "APM", "APL", "P", "R"]
STAGES = [
    ("T1", "t1_rat_initial"),
    ("T4", "t4_rat_no_amplitude"),
    ("T5", "t5_rat_channel_only"),
    ("T6", "t6_rat_amplitude_only"),
    ("T2", "t2_rat_late"),
    ("T3", "t3_rat_all"),
    ("T7", "t7_faar_rat_initial"),
]

STRUCTURES = {
    "D0": {
        "yaml": "yolov13-original.yaml（历史基线汇总）",
        "parent": "Original YOLOv13",
        "detail": "原始三尺度 YOLOv13；7 个融合隧道均为 FullPAD_Tunnel；两处自顶向下上采样均为 nearest；Detect(P3,P4,P5)。",
    },
    "B3": {
        "yaml": "yolov13_faar.yaml（历史 B3 汇总）",
        "parent": "Original YOLOv13",
        "detail": "在 D0 上将 P5→P4 上采样替换为 semantic FAARUp、P4→P3 替换为 detail FAARUp；7 个融合隧道仍为 FullPAD_Tunnel；Detect(P3,P4,P5)。",
    },
    "T1": {
        "yaml": "yolov13_rat_initial.yaml",
        "parent": "D0",
        "detail": "仅将 HyperACE 后最初 3 个 FullPAD_Tunnel 替换为 RATunnel；P4/P3/P5 均启用幅值对齐与通道门控，P4 额外启用空间门控；后续 4 个隧道保持 FullPAD；nearest 上采样；Detect(P3,P4,P5)。",
        "complexity": "2,572,676 参数，6.3 GFLOPs",
    },
    "T4": {
        "yaml": "yolov13_rat_no_amplitude.yaml",
        "parent": "D0",
        "detail": "T1 的去幅值对齐消融：最初 3 个 RATunnel 的 amplitude=False；通道门控保留，P4 空间门控保留；其余结构与 T1 相同。",
        "complexity": "2,572,676 参数，6.3 GFLOPs",
    },
    "T5": {
        "yaml": "yolov13_rat_channel_only.yaml",
        "parent": "D0",
        "detail": "T1 的去空间门控消融：最初 3 个 RATunnel 均保留幅值对齐与通道门控，但 spatial=False；其余结构与 T1 相同。",
        "complexity": "2,572,195 参数，6.3 GFLOPs",
    },
    "T6": {
        "yaml": "yolov13_rat_amplitude_only.yaml",
        "parent": "D0",
        "detail": "最初 3 个 RATunnel 仅保留幅值对齐与有界残差标量，channel=False、spatial=False；后续 4 个隧道为 FullPAD；其余结构与 D0 相同。",
        "complexity": "2,448,675 参数，6.2 GFLOPs",
    },
    "T2": {
        "yaml": "yolov13_rat_late.yaml",
        "parent": "D0",
        "detail": "最初 3 个隧道保持 FullPAD，仅将后续 4 个重复注入节点替换为 RATunnel；P4 节点启用幅值/通道/空间，P3/P5 启用幅值/通道；Detect(P3,P4,P5)。",
        "complexity": "2,578,357 参数，6.3 GFLOPs",
    },
    "T3": {
        "yaml": "yolov13_rat_all.yaml",
        "parent": "D0",
        "detail": "全部 7 个 FullPAD_Tunnel 均替换为 RATunnel；P4 节点启用幅值/通道/空间，P3/P5 节点启用幅值/通道；Detect(P3,P4,P5)。",
        "complexity": "2,702,358 参数，6.4 GFLOPs",
    },
    "T7": {
        "yaml": "yolov13_faar_rat_initial.yaml",
        "parent": "B3",
        "detail": "在 B3 双 FAARUp 基础上，将 HyperACE 后最初 3 个 FullPAD_Tunnel 替换为与 T1 相同的 RATunnel；后续 4 个隧道保持 FullPAD；Detect(P3,P4,P5)。",
        "complexity": "2,769,286 参数，7.3 GFLOPs",
    },
}


def remote_cmd(client: paramiko.SSHClient, script: str, timeout: int = 120) -> tuple[str, str]:
    encoded = base64.b64encode(script.encode("utf-8")).decode("ascii")
    command = f"bash -lc 'printf %s {encoded} | base64 -d | bash'"
    _stdin, stdout, stderr = client.exec_command(command, timeout=timeout)
    return stdout.read().decode("utf-8", "replace"), stderr.read().decode("utf-8", "replace")


def read_json(sftp: paramiko.SFTPClient, root: str, relative: str) -> dict[str, Any]:
    with sftp.file(posixpath.join(root, relative), "r") as handle:
        return json.loads(handle.read().decode("utf-8", "replace"))


def first_json(sftp: paramiko.SFTPClient, root: str, candidates: list[str]) -> tuple[dict[str, Any], str]:
    errors: list[str] = []
    for relative in candidates:
        try:
            return read_json(sftp, root, relative), relative
        except (FileNotFoundError, OSError, json.JSONDecodeError) as exc:
            errors.append(f"{relative}: {exc}")
    raise FileNotFoundError("No usable JSON found:\n" + "\n".join(errors))


def normalize(data: dict[str, Any], label: str) -> dict[str, Any]:
    rows = data.get("rows_percent") or data.get("rows") or []
    if len(rows) != 3:
        raise ValueError(f"{label} expected 3 seed rows, found {len(rows)}")
    cleaned: list[dict[str, Any]] = []
    for index, row in enumerate(sorted(rows, key=lambda item: int(item.get("seed", 0)))):
        clean = {"seed": int(row.get("seed", index))}
        for metric in METRICS:
            clean[metric] = float(row[metric])
            if not math.isfinite(clean[metric]):
                raise ValueError(f"{label} seed {clean['seed']} has non-finite {metric}")
        cleaned.append(clean)
    mean = {metric: statistics.fmean(row[metric] for row in cleaned) for metric in METRICS}
    std = {metric: statistics.stdev(row[metric] for row in cleaned) for metric in METRICS}
    return {
        "label": label,
        "rows": cleaned,
        "mean": mean,
        "std": std,
        "range": {metric: max(row[metric] for row in cleaned) - min(row[metric] for row in cleaned) for metric in METRICS},
    }


def delta(stage: dict[str, Any], base: dict[str, Any], metric: str) -> float:
    return stage["mean"][metric] - base["mean"][metric]


def paired(stage: dict[str, Any], base: dict[str, Any], metric: str) -> list[float]:
    base_rows = {row["seed"]: row for row in base["rows"]}
    return [row[metric] - base_rows[row["seed"]][metric] for row in stage["rows"]]


def wins(stage: dict[str, Any], base: dict[str, Any], metric: str) -> int:
    return sum(value > 0 for value in paired(stage, base, metric))


def fmt(value: float, signed: bool = False) -> str:
    return f"{value:+.3f}" if signed else f"{value:.3f}"


def metric_values(values: dict[str, float], signed: bool = False) -> str:
    return " | ".join(fmt(values[metric], signed) for metric in METRICS)


def load_payload(args: argparse.Namespace) -> dict[str, Any]:
    client = paramiko.SSHClient()
    client.set_missing_host_key_policy(paramiko.AutoAddPolicy())
    client.connect(
        args.host,
        port=args.port,
        username=args.user,
        password=args.password,
        timeout=20,
        banner_timeout=20,
        auth_timeout=20,
    )
    status_script = f"""
cd {args.root}
printf '%s\n' '===RAT_STATE==='
cat runs/rat_ablation/state.json 2>/dev/null || true
printf '%s\n' '===ACTIVE_RAT_PROJECT_PROCESSES==='
ps -u {args.user} -o pid,ppid,stat,etime,cmd | grep -E 'rat_ablation|train_rat_worker|rat_.*seed|yolov13_faar_rat|yolov13_rat_' | grep -v -E 'grep|bash -lc' || true
printf '%s\n' '===PROJECT_GPU_PROCESSES==='
for pid in $(nvidia-smi --query-compute-apps=pid --format=csv,noheader,nounits 2>/dev/null); do
  user=$(ps -o user= -p "$pid" 2>/dev/null | xargs)
  cwd=$(readlink -f "/proc/$pid/cwd" 2>/dev/null || true)
  cmd=$(xargs -0 < "/proc/$pid/cmdline" 2>/dev/null || true)
  case "$cwd $cmd" in *yolov13-305*) printf '%s|%s|%s|%s\n' "$pid" "$user" "$cwd" "$cmd" ;; esac
done
printf '%s\n' '===RAT_INVENTORY==='
find runs/train -maxdepth 1 -type d -name 'rat_*' | wc -l
find runs/test -maxdepth 1 -type d -name 'rat_*' | wc -l
find runs/test -maxdepth 1 -type f -name 'rat_*summary.json' | wc -l
find runs/rat_ablation -maxdepth 1 -type f 2>/dev/null | wc -l
"""
    status_out, status_err = remote_cmd(client, status_script)
    sftp = client.open_sftp()
    state, state_path = first_json(sftp, args.root, ["runs/rat_ablation/state.json"])
    run_id = str(state.get("run_id") or args.run_id or "").strip()
    if not run_id:
        raise ValueError("RAT state does not contain run_id")

    d0_raw, d0_path = first_json(
        sftp,
        args.root,
        [
            "runs/test/baseline_d0_original_20260708_215137_summary.json",
            "runs/test/csrfa_20260708_215137_d0_original_summary.json",
        ],
    )
    b3_raw, b3_path = first_json(
        sftp,
        args.root,
        ["runs/test/faar_20260708_155355_b3_scale_specific_summary.json"],
    )
    groups: dict[str, dict[str, Any]] = {"D0": normalize(d0_raw, "D0"), "B3": normalize(b3_raw, "B3")}
    sources = {"D0": d0_path, "B3": b3_path, "state": state_path}
    for label, stage in STAGES:
        raw, path = first_json(
            sftp,
            args.root,
            [
                f"runs/test/rat_{run_id}_{stage}_summary.json",
                f"runs/test/{run_id}_{stage}_summary.json",
            ],
        )
        groups[label] = normalize(raw, label)
        sources[label] = path
    sftp.close()
    client.close()
    return {
        "run_id": run_id,
        "state": state,
        "remote_status": status_out,
        "remote_status_stderr": status_err,
        "groups": groups,
        "sources": sources,
    }


def generate_report(payload: dict[str, Any]) -> str:
    groups = payload["groups"]
    d0, b3 = groups["D0"], groups["B3"]
    labels = ["D0", "B3", "T1", "T4", "T5", "T6", "T2", "T3", "T7"]
    rat_labels = ["T1", "T4", "T5", "T6", "T2", "T3", "T7"]
    lines: list[str] = []
    lines.append("RAT-YOLOv13 三次重复实验数据详细分析报告")
    lines.append(f"生成时间：{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    lines.append(f"RAT 链式实验 Run ID：{payload['run_id']}")
    lines.append("用途：为后续 GPT/科研分析提供结构—指标—稳定性对应证据。")
    lines.append("")
    lines.append("一、分析口径与完整性")
    lines.append("")
    lines.append("1. 全部指标来自服务器独立测试汇总 JSON，单位均为百分比；每个配置固定纳入 seed0、seed1、seed2 三次结果。")
    lines.append("2. 均值与样本标准差由三次原始值重新计算；同时检查逐 seed 同号性、最好/最差、极差以及相对同 seed 对照的胜出次数。")
    lines.append("3. T1/T4/T5/T6/T2/T3 的母体为 D0，因此主比较对象为原始 YOLOv13；T7 的母体为 B3，因此主比较对象为 B3，同时报告所有配置相对 D0 与 B3 的差值。")
    lines.append("4. 样本量仅 n=3，本报告做描述性与重复性判断，不把很小的均值差或单个 seed 峰值解释为统计显著提升。")
    lines.append("5. 所有配置保持相同三尺度 Detect(P3,P4,P5)、训练环境、数据集、损失函数与分配策略；变量仅为 RATunnel 的位置/子组件以及 T7 的 FAARUp 母体。")
    lines.append("")
    lines.append("远程状态核验原文：")
    lines.append(payload["remote_status"].strip() or "（无输出）")
    if payload.get("remote_status_stderr", "").strip():
        lines.append("远程状态 stderr：")
        lines.append(payload["remote_status_stderr"].strip())
    lines.append("")
    lines.append("二、实验编号与结构配置对应关系")
    lines.append("")
    lines.append("| 编号 | YAML | 母体 | 结构配置 | 复杂度（自检值） |")
    lines.append("|---|---|---|---|---|")
    for label in labels:
        info = STRUCTURES[label]
        lines.append(f"| {label} | {info['yaml']} | {info['parent']} | {info['detail']} | {info.get('complexity', '历史汇总未重算')} |")
    lines.append("")
    lines.append("RATunnel 公共机制：输入顺序为 [original, enhanced]，输出通道/空间尺寸与 original 一致；enhanced 先投影，幅值比采用 detach 后的 RMS 比值并限制在 [0.5, 2.0]，通道/空间可靠性门控按配置启用；gamma_raw 零初始化并通过有界残差注入，使初始行为接近恒等映射。P4 的 gamma 上限 0.15，P3/P5 为 0.10。")
    lines.append("")
    lines.append("三、三 seed 总体结果（均值±样本标准差）")
    lines.append("")
    lines.append("| 组别 | " + " | ".join(METRICS) + " |")
    lines.append("|---|" + "|".join(["---:"] * len(METRICS)) + "|")
    for label in labels:
        result = groups[label]
        values = " | ".join(f"{result['mean'][metric]:.3f}±{result['std'][metric]:.3f}" for metric in METRICS)
        lines.append(f"| {label} | {values} |")
    lines.append("")
    ranking = sorted(labels, key=lambda label: groups[label]["mean"]["mAP50-95"], reverse=True)
    lines.append("核心 mAP50-95 均值排序：" + " > ".join(f"{label} {groups[label]['mean']['mAP50-95']:.3f}" for label in ranking) + "。")
    lines.append("")
    lines.append("四、相对 D0 与 B3 的均值差（百分点）")
    lines.append("")
    lines.append("| 组别 | 对照 | " + " | ".join(METRICS) + " |")
    lines.append("|---|---|" + "|".join(["---:"] * len(METRICS)) + "|")
    for label in rat_labels:
        for base_label in ["D0", "B3"]:
            diffs = {metric: delta(groups[label], groups[base_label], metric) for metric in METRICS}
            lines.append(f"| {label} | {base_label} | {metric_values(diffs, signed=True)} |")
    lines.append("")
    lines.append("五、逐 seed 原始结果")
    lines.append("")
    for label in labels:
        lines.append(f"{label}：{STRUCTURES[label]['detail']}")
        lines.append("")
        lines.append("| Seed | " + " | ".join(METRICS) + " |")
        lines.append("|---:|" + "|".join(["---:"] * len(METRICS)) + "|")
        for row in groups[label]["rows"]:
            lines.append(f"| {row['seed']} | {metric_values(row)} |")
        lines.append("")
    lines.append("六、逐 seed 一致性、极差与稳定性")
    lines.append("")
    for label in rat_labels:
        base_label = "B3" if label == "T7" else "D0"
        stage, base = groups[label], groups[base_label]
        lines.append(f"{label} 相对其主对照 {base_label}：")
        for metric in CORE_METRICS:
            changes = paired(stage, base, metric)
            lines.append(
                f"- {metric}：均值差 {delta(stage, base, metric):+.3f}；逐 seed 差 [{', '.join(fmt(value, True) for value in changes)}]；"
                f"胜出 {sum(value > 0 for value in changes)}/3；{label} std={stage['std'][metric]:.3f}、极差={stage['range'][metric]:.3f}。"
            )
        lines.append("")
    lines.append("七、RATunnel 组件与位置消融读数")
    lines.append("")
    comparisons = [
        ("幅值对齐的边际作用", "T1", "T4", "T1−T4；两者都保留通道门控与 P4 空间门控"),
        ("空间门控的边际作用", "T1", "T5", "T1−T5；两者都保留幅值对齐与通道门控"),
        ("通道+空间可靠性门控的合并作用", "T1", "T6", "T1−T6；两者都保留幅值对齐"),
        ("初始三节点相对后期四节点", "T1", "T2", "T1−T2；注意位置与节点数量同时不同"),
        ("全七节点相对初始三节点", "T3", "T1", "T3−T1；衡量新增后期四节点"),
        ("全七节点相对后期四节点", "T3", "T2", "T3−T2；衡量新增初始三节点"),
        ("B3 上增加初始 RAT", "T7", "B3", "T7−B3"),
        ("T1 上增加双 FAARUp", "T7", "T1", "T7−T1"),
    ]
    lines.append("| 消融问题 | 差值定义 | mAP50-95 | mAP75 | mAP50 | APS | APM | APL | P | R |")
    lines.append("|---|---|---:|---:|---:|---:|---:|---:|---:|---:|")
    for title, left, right, definition in comparisons:
        values = {metric: delta(groups[left], groups[right], metric) for metric in METRICS}
        lines.append(
            f"| {title} | {definition} | {values['mAP50-95']:+.3f} | {values['mAP75']:+.3f} | {values['mAP50']:+.3f} | "
            f"{values['APS']:+.3f} | {values['APM']:+.3f} | {values['APL']:+.3f} | {values['P']:+.3f} | {values['R']:+.3f} |"
        )
    lines.append("")
    for title, left, right, definition in comparisons:
        changes = paired(groups[left], groups[right], "mAP50-95")
        lines.append(
            f"- {title}（{definition}）：mAP50-95 均值差 {delta(groups[left], groups[right], 'mAP50-95'):+.3f}，"
            f"逐 seed 差 [{', '.join(fmt(value, True) for value in changes)}]，正向 {sum(value > 0 for value in changes)}/3。"
        )
    lines.append("")
    lines.append("八、逐实验判定")
    lines.append("")
    for label in rat_labels:
        base_label = "B3" if label == "T7" else "D0"
        stage, base = groups[label], groups[base_label]
        core_delta = delta(stage, base, "mAP50-95")
        core_wins = wins(stage, base, "mAP50-95")
        positive_metrics = [metric for metric in METRICS if delta(stage, base, metric) > 0]
        negative_metrics = [metric for metric in METRICS if delta(stage, base, metric) < 0]
        if core_delta > 0 and core_wins == 3:
            verdict = "mAP50-95 在三次同 seed 比较中方向一致为正，可视为可重复的正向信号；仍需结合增益幅度和其他指标取舍。"
        elif core_delta > 0 and core_wins == 2:
            verdict = "mAP50-95 均值为正但仅 2/3 seed 胜出，属于有限且不完全稳定的正向信号。"
        elif core_delta > 0:
            verdict = "mAP50-95 均值略正但逐 seed 支持不足，不能认定为可靠提升。"
        else:
            verdict = "mAP50-95 均值未超过主对照，不能认定为有效总体提升。"
        lines.append(f"{label}（主对照 {base_label}）：")
        lines.append(
            f"- mAP50-95={stage['mean']['mAP50-95']:.3f}±{stage['std']['mAP50-95']:.3f}，"
            f"相对 {base_label} {core_delta:+.3f}，逐 seed 胜出 {core_wins}/3。{verdict}"
        )
        lines.append(f"- 均值高于主对照的指标：{', '.join(positive_metrics) or '无'}；低于主对照的指标：{', '.join(negative_metrics) or '无'}。")
        lines.append("")
    lines.append("九、面向下一步 GPT 分析的事实约束与总判断")
    lines.append("")
    best_rat = max(rat_labels, key=lambda label: groups[label]["mean"]["mAP50-95"])
    lines.append(
        f"1. 本轮 RAT 配置中 mAP50-95 均值最高的是 {best_rat}（{groups[best_rat]['mean']['mAP50-95']:.3f}±{groups[best_rat]['std']['mAP50-95']:.3f}）；"
        f"D0 为 {d0['mean']['mAP50-95']:.3f}±{d0['std']['mAP50-95']:.3f}，B3 为 {b3['mean']['mAP50-95']:.3f}±{b3['std']['mAP50-95']:.3f}。"
    )
    lines.append(
        "2. 判断优先级应是：先看 mAP50-95 相对正确母体的三 seed 同号性，再看 mAP75 与 APS/APM/APL 是否同步，最后才参考单次最高值；任何只有 1/3 seed 支撑的峰值都不应作为下一步方案的主要依据。"
    )
    t7_b3 = delta(groups["T7"], b3, "mAP50-95")
    lines.append(
        f"3. T7 是唯一能直接回答 RAT 与 B3 是否兼容的实验：其相对 B3 的 mAP50-95 均值差为 {t7_b3:+.3f}，"
        f"逐 seed 差为 [{', '.join(fmt(value, True) for value in paired(groups['T7'], b3, 'mAP50-95'))}]。"
    )
    lines.append(
        "4. T1/T4/T5/T6 的互比用于识别幅值、通道和空间分支的贡献；T1/T2/T3 用于识别注入阶段与堆叠数量。由于部分对比同时改变节点数量或母体，下一步 GPT 不应把这些结果解释成严格单变量因果。"
    )
    lines.append(
        "5. 所有结论均基于相同测试口径的三次重复，但 n=3 仍然很小。若某配置的均值增益小于其自身或对照的 seed 波动，报告只将其称为趋势，而不使用“显著提升”。"
    )
    lines.append(
        f"6. 没有任何 RAT 配置超过 B3 的 mAP50-95 均值。RAT 中最高的 T5 仍比 B3 低 "
        f"{abs(delta(groups['T5'], b3, 'mAP50-95')):.3f}；因此本轮数据不支持 RATunnel 已取得优于现有 B3 的总体检测精度。"
    )
    lines.append(
        f"7. T5 相对 D0 的 mAP50-95 均值为 {delta(groups['T5'], d0, 'mAP50-95'):+.3f}，但逐 seed 差为 "
        f"[{', '.join(fmt(value, True) for value in paired(groups['T5'], d0, 'mAP50-95'))}]：只有 seed1 明显为正，seed0/seed2 均略负。"
        "这说明 T5 的均值优势由单次结果主导，而不是三次一致复现。"
    )
    lines.append(
        f"8. RAT 初始节点对大目标存在局部一致信号：T1 的 APL 相对 D0 为 {delta(groups['T1'], d0, 'APL'):+.3f} 且 3/3 seed 为正，"
        f"T5 的 APL 为 {delta(groups['T5'], d0, 'APL'):+.3f} 且同样 3/3 为正；但两者 APS 分别为 "
        f"{delta(groups['T1'], d0, 'APS'):+.3f}、{delta(groups['T5'], d0, 'APS'):+.3f}，表明收益偏向大目标而非全尺度改善。"
    )
    lines.append(
        f"9. 幅值对齐没有显示净总体收益：T1 相比去幅值版本 T4 的 mAP50-95 为 {delta(groups['T1'], groups['T4'], 'mAP50-95'):+.3f}，"
        f"逐 seed 仅 {wins(groups['T1'], groups['T4'], 'mAP50-95')}/3 为正。空间门控同样没有得到支持：T1 相比去空间版本 T5 的 "
        f"mAP50-95 为 {delta(groups['T1'], groups['T5'], 'mAP50-95'):+.3f}、mAP75 为 {delta(groups['T1'], groups['T5'], 'mAP75'):+.3f}。"
    )
    lines.append(
        f"10. 全 7 节点堆叠 T3 是方向最明确的负结果：相对 D0 的 mAP50-95 为 {delta(groups['T3'], d0, 'mAP50-95'):+.3f}，"
        f"三个 seed 全部下降；mAP75 为 {delta(groups['T3'], d0, 'mAP75'):+.3f}，同样 0/3 胜出。数据不支持扩大 RAT 注入范围。"
    )
    lines.append(
        f"11. T7 呈现稳定的精确率—召回率/小目标权衡：相对 B3，P {delta(groups['T7'], b3, 'P'):+.3f} 且 3/3 上升，"
        f"mAP75 {delta(groups['T7'], b3, 'mAP75'):+.3f} 且 3/3 上升；但 R {delta(groups['T7'], b3, 'R'):+.3f}、"
        f"APS {delta(groups['T7'], b3, 'APS'):+.3f}，最终 mAP50-95 {delta(groups['T7'], b3, 'mAP50-95'):+.3f} 且 0/3 超过 B3。"
    )
    lines.append(
        "12. 总判定：RATunnel 当前版本没有形成可作为新主干方案的稳定综合提升。可确认的只是两类局部现象——初始节点配置对 APL 的一致正向偏置，"
        "以及与 B3 组合后对 P/mAP75 的一致提升；两者都伴随 APS、R 或总体 mAP50-95 的代价，不能等价为模型整体有效。"
    )
    lines.append("")
    lines.append("十、数据来源清单")
    lines.append("")
    for label in ["D0", "B3", *rat_labels]:
        lines.append(f"- {label}: {payload['sources'][label]}")
    lines.append(f"- 链式状态: {payload['sources']['state']}")
    lines.append("")
    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--host", default="154.9.253.153")
    parser.add_argument("--port", type=int, default=29103)
    parser.add_argument("--user", default="rom305")
    parser.add_argument("--password", default=os.environ.get("YOLO_SSH_PASSWORD", "Room305@!"))
    parser.add_argument("--root", default="/home/rom305/zzf/yolov13-305")
    parser.add_argument("--run-id", default="")
    parser.add_argument("--output", type=Path, default=Path(f"RAT_YOLOv13_实验数据详细分析报告_{datetime.now():%Y%m%d}.txt"))
    parser.add_argument("--payload-output", type=Path, default=Path(".codex_tmp/rat_analysis_payload.json"))
    args = parser.parse_args()

    payload = load_payload(args)
    args.payload_output.parent.mkdir(parents=True, exist_ok=True)
    args.payload_output.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    args.output.write_text(generate_report(payload), encoding="utf-8")
    print(args.output.resolve())
    print(args.payload_output.resolve())


if __name__ == "__main__":
    main()
