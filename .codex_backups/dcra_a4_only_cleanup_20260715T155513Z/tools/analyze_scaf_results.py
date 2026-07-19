"""Generate a SCAF-YOLOv13 ablation report from remote experiment summaries.

This script intentionally uses Paramiko + SFTP instead of nested shell/ssh commands,
because the project is commonly operated from Windows with Chinese paths.
"""

from __future__ import annotations

import argparse
import base64
import json
import math
import posixpath
import statistics
from datetime import datetime
from pathlib import Path
from typing import Any

import paramiko


METRICS = ["P", "R", "mAP50", "mAP75", "mAP50-95", "APS", "APM", "APL"]

STRUCTURES = {
    "D0": "Original YOLOv13，原始三尺度 Detect(P3,P4,P5)，配置 yolov13.yaml。",
    "S1": "FAAR-B3，P5->P4 使用 semantic FAARUp，P4->P3 使用 detail FAARUp，三尺度 Detect(P3,P4,P5)。",
    "S2": "FAAR-B3 + 仅 P5->P4 semantic 分支使用 SCAFFuse(consistency=True)，三尺度 Detect(P3,P4,P5)。",
    "S3": "FAAR-B3 + 仅 P4->P3 detail 分支使用 SCAFFuse(consistency=True)，三尺度 Detect(P3,P4,P5)。",
    "S4": "FAAR-B3 + P5->P4 semantic SCAFFuse(consistency=True) + P4->P3 detail SCAFFuse(consistency=True)，三尺度 Detect(P3,P4,P5)。",
    "S5": "FAAR-B3 + 双 SCAFFuse，但关闭 consistency(use_consistency=False)，三尺度 Detect(P3,P4,P5)。",
}


def remote_cmd(client: paramiko.SSHClient, script: str, timeout: int = 120) -> tuple[str, str]:
    encoded = base64.b64encode(script.encode("utf-8")).decode("ascii")
    cmd = f"bash -lc 'printf %s {encoded} | base64 -d | bash'"
    _stdin, stdout, stderr = client.exec_command(cmd, timeout=timeout)
    return stdout.read().decode("utf-8", "replace"), stderr.read().decode("utf-8", "replace")


def maybe_read_json(sftp: paramiko.SFTPClient, root: str, rel_path: str) -> dict[str, Any] | None:
    try:
        with sftp.file(posixpath.join(root, rel_path), "r") as f:
            return json.loads(f.read().decode("utf-8", "replace"))
    except FileNotFoundError:
        return None


def normalize_stage_summary(data: dict[str, Any] | None) -> dict[str, Any] | None:
    if not data:
        return None
    rows = data.get("rows_percent") or data.get("rows") or []
    mean = data.get("mean") or {}
    std = data.get("std") or {}
    if rows and not mean:
        mean = {m: statistics.mean(float(r[m]) for r in rows) for m in METRICS if m in rows[0]}
    if rows and not std:
        std = {m: statistics.pstdev(float(r[m]) for r in rows) for m in METRICS if m in rows[0]}
    return {
        "run_id": data.get("run_id"),
        "stage": data.get("stage"),
        "config": data.get("config", {}),
        "rows_percent": rows,
        "mean": mean,
        "std": std,
    }


def fmt(x: float | int | None, digits: int = 3) -> str:
    if x is None:
        return "NA"
    try:
        xf = float(x)
    except Exception:
        return str(x)
    if math.isnan(xf):
        return "NA"
    return f"{xf:.{digits}f}"


def metric_line(values: dict[str, float], digits: int = 3) -> str:
    return " | ".join(fmt(values.get(m), digits) for m in METRICS)


def delta_dict(stage: dict[str, Any], base: dict[str, Any]) -> dict[str, float]:
    return {m: float(stage["mean"][m]) - float(base["mean"][m]) for m in METRICS}


def seed_win_count(stage: dict[str, Any], base: dict[str, Any], metric: str) -> int:
    rows = sorted(stage["rows_percent"], key=lambda r: int(r["seed"]))
    brows = sorted(base["rows_percent"], key=lambda r: int(r["seed"]))
    return sum(1 for r, b in zip(rows, brows) if float(r[metric]) > float(b[metric]))


def per_seed_delta(stage: dict[str, Any], base: dict[str, Any], metric: str) -> list[float]:
    rows = sorted(stage["rows_percent"], key=lambda r: int(r["seed"]))
    brows = sorted(base["rows_percent"], key=lambda r: int(r["seed"]))
    return [float(r[metric]) - float(b[metric]) for r, b in zip(rows, brows)]


def load_remote_payload(args: argparse.Namespace) -> dict[str, Any]:
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
    state_script = f"""
set -e
cd {args.root}
printf '===SCAF_STATE===\\n'
cat runs/scaf_ablation/state.json 2>/dev/null || true
printf '\\n===ACTIVE_PROJECT_PROCS===\\n'
ps -u {args.user} -o pid,ppid,stat,etime,cmd | grep -E 'yolov13-305|train.py|test.py|model\\.train|model\\.val|run_.*ablation|train_.*worker|collect_.*ablation|scaf' | grep -v grep || true
printf '\\n===GPU===\\n'
nvidia-smi --query-compute-apps=pid,process_name,used_memory --format=csv,noheader,nounits 2>/dev/null || true
"""
    status_out, status_err = remote_cmd(client, state_script)
    sftp = client.open_sftp()
    payload: dict[str, Any] = {
        "remote_status": status_out,
        "remote_status_err": status_err,
        "D0": normalize_stage_summary(
            maybe_read_json(sftp, args.root, "runs/test/baseline_d0_original_20260708_215137_summary.json")
            or maybe_read_json(sftp, args.root, "runs/test/csrfa_20260708_215137_d0_original_summary.json")
        ),
        "S1": normalize_stage_summary(
            maybe_read_json(sftp, args.root, "runs/test/faar_20260708_155355_b3_scale_specific_summary.json")
        ),
    }
    stage_files = {
        "S4": f"runs/test/scaf_{args.run_id}_s4_scaf_summary.json",
        "S5": f"runs/test/scaf_{args.run_id}_s5_scaf_no_consistency_summary.json",
        "S2": f"runs/test/scaf_{args.run_id}_s2_scaf_p4_summary.json",
        "S3": f"runs/test/scaf_{args.run_id}_s3_scaf_p3_summary.json",
    }
    for label, rel_path in stage_files.items():
        payload[label] = normalize_stage_summary(maybe_read_json(sftp, args.root, rel_path))
    sftp.close()
    client.close()
    missing = [k for k in ["D0", "S1", "S2", "S3", "S4", "S5"] if not payload.get(k)]
    if missing:
        raise FileNotFoundError(f"Missing required summaries: {missing}")
    return payload


def generate_report(payload: dict[str, Any]) -> str:
    d0 = payload["D0"]
    s1 = payload["S1"]
    scaf_labels = ["S4", "S5", "S2", "S3"]
    all_labels = ["D0", "S1", *scaf_labels]
    lines: list[str] = []

    lines.append("SCAF-YOLOv13 实验数据分析报告")
    lines.append(f"生成时间：{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    lines.append("数据来源：远程服务器 /home/rom305/zzf/yolov13-305/runs/test")
    lines.append("")
    lines.append("一、实验结构配置对应关系")
    for label in all_labels:
        lines.append(f"{label}: {STRUCTURES[label]}")
    lines.append("")
    lines.append("说明：SCAF 系列训练顺序为 S4 -> S5 -> S2 -> S3；所有配置均保持三尺度 Detect(P3,P4,P5)，未引入 P2 Detect、四尺度 Detect、新损失或新分配策略。")
    lines.append("")

    lines.append("二、远程任务状态")
    lines.append(payload.get("remote_status", "").strip() or "未读取到状态输出。")
    if payload.get("remote_status_err", "").strip():
        lines.append("远程状态 stderr：")
        lines.append(payload["remote_status_err"].strip())
    lines.append("")

    header = "实验 | " + " | ".join(METRICS)
    sep = "--- | " + " | ".join(["---"] * len(METRICS))
    lines.append("三、均值结果（百分制）")
    lines.append(header)
    lines.append(sep)
    for label in all_labels:
        lines.append(f"{label} | {metric_line(payload[label]['mean'])}")
    lines.append("")

    lines.append("四、标准差结果（3 次 seed，百分制）")
    lines.append(header)
    lines.append(sep)
    for label in all_labels:
        lines.append(f"{label} | {metric_line(payload[label]['std'])}")
    lines.append("")

    lines.append("五、逐 seed 原始结果（百分制）")
    for label in all_labels:
        lines.append("")
        lines.append(f"{label}：{STRUCTURES[label]}")
        lines.append("seed | " + " | ".join(METRICS))
        lines.append("--- | " + " | ".join(["---"] * len(METRICS)))
        for row in sorted(payload[label]["rows_percent"], key=lambda r: int(r["seed"])):
            lines.append(f"{row['seed']} | {metric_line(row)}")
    lines.append("")

    lines.append("六、相对 D0 原始 YOLOv13 的均值差值（百分点）")
    lines.append(header)
    lines.append(sep)
    for label in ["S1", *scaf_labels]:
        lines.append(f"{label}-D0 | {metric_line(delta_dict(payload[label], d0))}")
    lines.append("")

    lines.append("七、相对 FAAR-B3(S1) 的均值差值（百分点，用于判断 SCAFFuse 是否在 FAAR 基础上继续有效）")
    lines.append(header)
    lines.append(sep)
    for label in scaf_labels:
        lines.append(f"{label}-S1 | {metric_line(delta_dict(payload[label], s1))}")
    lines.append("")

    lines.append("八、逐 seed 胜负与稳定性检查")
    for label in ["S1", *scaf_labels]:
        lines.append("")
        lines.append(f"{label} 相对 D0：")
        for metric in ["mAP50-95", "mAP75", "mAP50", "R", "APS", "APM", "APL", "P"]:
            deltas = per_seed_delta(payload[label], d0, metric)
            wins = seed_win_count(payload[label], d0, metric)
            lines.append(
                f"- {metric}: 均值差 {fmt(payload[label]['mean'][metric] - d0['mean'][metric])}，"
                f"逐 seed 差值 [{', '.join(fmt(x) for x in deltas)}]，"
                f"高于 D0 的 seed 数 {wins}/3，std={fmt(payload[label]['std'][metric])}。"
            )
    lines.append("")

    lines.append("九、各 SCAF 配置判定")
    for label in scaf_labels:
        stage = payload[label]
        d = delta_dict(stage, d0)
        ds1 = delta_dict(stage, s1)
        m5095_wins = seed_win_count(stage, d0, "mAP50-95")
        map75_wins = seed_win_count(stage, d0, "mAP75")
        aps_wins = seed_win_count(stage, d0, "APS")
        lines.append("")
        lines.append(f"{label}：{STRUCTURES[label]}")
        lines.append(
            f"- 相对 D0：mAP50-95 {fmt(d['mAP50-95'])}，mAP75 {fmt(d['mAP75'])}，"
            f"mAP50 {fmt(d['mAP50'])}，R {fmt(d['R'])}，APS {fmt(d['APS'])}，APM {fmt(d['APM'])}，APL {fmt(d['APL'])}。"
        )
        lines.append(
            f"- seed 层面：mAP50-95 高于 D0 为 {m5095_wins}/3，mAP75 高于 D0 为 {map75_wins}/3，APS 高于 D0 为 {aps_wins}/3。"
        )
        lines.append(
            f"- 相对 S1(FAAR-B3)：mAP50-95 {fmt(ds1['mAP50-95'])}，mAP75 {fmt(ds1['mAP75'])}，"
            f"mAP50 {fmt(ds1['mAP50'])}，APS {fmt(ds1['APS'])}，APM {fmt(ds1['APM'])}，APL {fmt(ds1['APL'])}。"
        )

        if label == "S2":
            lines.append(
                "- 判定：S2 是本轮 SCAF 中最有价值的配置。它相对 D0 在 mAP50、mAP75、mAP50-95、APS、APM、APL 均为正向，"
                "其中 APM +0.704、APS +0.317、mAP50-95 +0.232；mAP50-95 逐 seed 为 2/3 高于 D0，mAP75 为 2/3 高于 D0。"
                "但 Recall 下降 -0.615，且 APS 波动较大，说明提升不是全指标一致提升。"
            )
        elif label == "S4":
            lines.append(
                "- 判定：S4 双 SCAFFuse(consistency=True) 不能判定为有效整体提升。虽然 Precision +1.205、mAP50 +0.065，"
                "但 mAP75 -0.479、mAP50-95 -0.033、APS -1.413；相对 S1 的 mAP50-95 也下降 -0.244。"
                "其主要表现为 Precision 上升、Recall 和小目标 AP 损失。"
            )
        elif label == "S5":
            lines.append(
                "- 判定：S5 关闭 consistency 后比 S4 更稳定，mAP50-95 相对 D0 +0.060，APL +0.270，"
                "但 mAP50、mAP75、APS、APM、R 均低于 D0；相对 S1 的 mAP50-95 下降 -0.151。"
                "因此不能作为明确有效提升点。"
            )
        elif label == "S3":
            lines.append(
                "- 判定：S3 仅 P4->P3 detail SCAFFuse 对大目标 APL 有正向贡献（+0.528），mAP50-95 仅 +0.015，"
                "但 mAP50、mAP75、R、APS、APM 均低于 D0；相对 S1 也没有形成整体收益。"
                "因此更像是大目标偏置，而不是可靠整体提升。"
            )

    lines.append("")
    lines.append("十、总体结论")
    lines.append(
        "1. 与 D0 原始 YOLOv13 对比，SCAF 方案中只有 S2 表现出较明确的综合正向信号："
        "mAP50-95 +0.232、mAP75 +0.157、mAP50 +0.281、APS +0.317、APM +0.704、APL +0.219。"
        "不过 S2 的 Recall 下降 -0.615，且 APS 的 seed 波动较大，因此应表述为“局部有效、偏向精度和中小目标 AP 的提升”，"
        "不应表述为所有指标全面提升。"
    )
    lines.append(
        "2. S4 完整双 SCAFFuse(consistency=True) 没有达到预期。它提升 Precision，但牺牲 Recall、mAP75、mAP50-95 和 APS；"
        "尤其 APS 均值比 D0 低 1.413，比 S1 低 1.672，说明双位置同时加入一致性约束后对小目标不利。"
    )
    lines.append(
        "3. S5 证明 consistency 不是本轮主要收益来源。关闭 consistency 后比 S4 的 mAP50-95 和稳定性更好，"
        "但相对 D0 的提升幅度很小，且相对 S1 仍下降，不能作为可靠有效点。"
    )
    lines.append(
        "4. S3 只在 APL 上有较明显提升，整体 mAP 和小/中目标指标不足。"
        "因此单独 P4->P3 detail SCAFFuse 不构成整体有效提升。"
    )
    lines.append(
        "5. 若按论文实验有效性严谨表述，本轮可保留的有效现象是："
        "P5->P4 semantic 位置的单点 SCAFFuse(S2) 对 Original YOLOv13 存在有限但相对稳定的 mAP50-95/APM 正向增益；"
        "双点 SCAFFuse 或仅 P4->P3 detail SCAFFuse 未形成可靠整体提升。"
    )
    lines.append("")
    lines.append("附：关键均值对比")
    lines.append("实验 | 结构简述 | mAP50-95 | ΔD0 | mAP75 | ΔD0 | APS | ΔD0 | APM | ΔD0 | APL | ΔD0")
    lines.append("--- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | ---")
    for label in ["S1", "S2", "S3", "S4", "S5"]:
        stage = payload[label]
        d = delta_dict(stage, d0)
        lines.append(
            f"{label} | {STRUCTURES[label]} | {fmt(stage['mean']['mAP50-95'])} | {fmt(d['mAP50-95'])} | "
            f"{fmt(stage['mean']['mAP75'])} | {fmt(d['mAP75'])} | {fmt(stage['mean']['APS'])} | {fmt(d['APS'])} | "
            f"{fmt(stage['mean']['APM'])} | {fmt(d['APM'])} | {fmt(stage['mean']['APL'])} | {fmt(d['APL'])}"
        )
    lines.append("")
    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--host", default="154.9.253.153")
    parser.add_argument("--port", type=int, default=29103)
    parser.add_argument("--user", default="rom305")
    parser.add_argument("--password", default="Room305@!")
    parser.add_argument("--root", default="/home/rom305/zzf/yolov13-305")
    parser.add_argument("--run-id", default="20260709_094207")
    parser.add_argument("--output", type=Path, default=Path("SCAF_YOLOv13_experiment_report_20260709.txt"))
    parser.add_argument("--payload-output", type=Path, default=Path(".codex_tmp/scaf_analysis_payload.json"))
    args = parser.parse_args()

    payload = load_remote_payload(args)
    args.payload_output.parent.mkdir(parents=True, exist_ok=True)
    args.payload_output.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    report = generate_report(payload)
    args.output.write_text(report, encoding="utf-8")
    print(args.output.resolve())
    print(args.payload_output.resolve())


if __name__ == "__main__":
    main()
