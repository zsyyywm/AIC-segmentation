"""Analyze one completed AIC run and append the result to result.md."""

import argparse
from pathlib import Path

from experiment_results import (append_result, fmt, load_run, report_time,
                                run_identity, summarize)


def model_text(model):
    if not model:
        return 'N/A'
    fields = []
    for key in ('type', 'backbone', 'decode_head', 'auxiliary_head'):
        if model.get(key):
            fields.append(f'{key}={model[key]}')
    return ', '.join(fields) if fields else str(model)


def make_report(run, stats):
    identity = run_identity(run)
    summary = run.summary
    lines = [
        f'## 单模型分析：{identity["run_id"]}', '',
        f'- 分析时间：`{report_time()}`',
        f'- 运行目录：`{run.directory}`',
        f'- 实验名称：`{identity["experiment"]}`',
        f'- 配置：`{identity["config"]}`',
        f'- 模型：{model_text(identity["model"])}',
        f'- 计划训练：{identity["max_iterations"]} iter，'
        f'{identity["equivalent_epochs"]} 等效 epoch',
        f'- Batch：物理 {identity["physical_batch"]}，累积 '
        f'{identity["accumulation"]}，有效 {identity["effective_batch"]}',
        f'- 随机种子：`{identity["seed"]}`',
        f'- 预训练：`{identity["pretrained"]}`',
        f'- 代表权重：`{stats["checkpoint"] or "N/A"}` '
        f'({stats["checkpoint_size_mb"] or "N/A"} MB)', '',
        '### 总体指标', '',
        '| 指标 | 数值 |', '|---|---:|',
        f'| 稳定窗口 | 最后 {stats["window_size"]} / '
        f'{stats["validation_count"]} 个验证节点 |',
        f'| 代表 mIoU（窗口中位数） | {fmt(stats["representative_miou"])} |',
        f'| 窗口 mIoU 均值 ± 标准差 | {fmt(stats["tail_mean"])} ± '
        f'{fmt(stats["tail_std"])} |',
        f'| 最终 mIoU | {fmt(stats["final_miou"])} |',
        f'| 最佳 mIoU | {fmt(stats["best_miou"])} @ iter '
        f'{stats["best_iter"]} |',
        f'| 最佳值与最终值差距 | {fmt(stats["best_final_gap"])} |',
        f'| 归一化 mIoU AUC | {fmt(stats["normalized_auc"])} |',
        f'| 窗口趋势（百分点/验证节点） | '
        f'{fmt(stats["tail_slope_per_validation"], 3)} |',
        f'| 首次 / 最后训练 loss | {fmt(stats["first_loss"], 4)} / '
        f'{fmt(stats["final_loss"], 4)} |',
        f'| 实际训练时间 | {fmt(summary.get("training_time_hours"))} h |',
        f'| 峰值预留显存 | {fmt(summary.get("peak_memory_reserved_mb"))} MB |',
        '', '### 各类别', '',
        '| 类别 | 窗口中位数 IoU | 最终 IoU | 窗口标准差 |',
        '|---|---:|---:|---:|',
    ]
    for name, values in stats['class_metrics'].items():
        lines.append(f'| {name} | {fmt(values["representative"])} | '
                     f'{fmt(values["final"])} | {fmt(values["std"])} |')
    lines.extend(['', '### 初步判断', '', f'- {stats["state"]}。'])
    if run.summary.get('finished') is not True:
        lines.append('- 训练未被日志确认完整结束，本条结果只能视为阶段性记录。')
    if run.warnings:
        lines.append('- 日志警告：' + '；'.join(run.warnings) + '。')
    lines.append('- 该结果来自固定本地验证集，不是官方测试集分数。')
    return '\n'.join(lines)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('source', help='Run directory or checkpoint inside it')
    args = parser.parse_args()
    try:
        run = load_run(args.source)
        checkpoint = (Path(args.source).resolve()
                      if Path(args.source).suffix.lower() == '.pth' else None)
        stats = summarize(run, checkpoint)
        report = make_report(run, stats)
        output = append_result(report)
        print(report)
        print(f'\nAPPENDED: {output}')
    except (OSError, ValueError) as exc:
        parser.exit(1, f'ERROR: {exc}\n')


if __name__ == '__main__':
    main()
