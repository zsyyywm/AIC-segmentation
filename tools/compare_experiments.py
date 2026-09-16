"""Compare two AIC runs and append the result to the experiment ledger."""

import argparse

from experiment_results import (CLASSES, append_result, fmt, load_run,
                                report_time, run_identity, summarize)


def completed_budget(run):
    summary, manifest = run.summary, run.manifest
    step = max(run.validations)
    physical = manifest.get('physical_batch_size')
    accumulation = manifest.get('gradient_accumulation_steps')
    return {
        'samples': summary.get('completed_sample_exposures',
                               step * physical if physical else None),
        'updates': summary.get('completed_optimizer_steps',
                               step // accumulation if accumulation else None),
    }


def fairness_checks(baseline, candidate):
    mismatches, unknown = [], []
    bm, cm = baseline.manifest, candidate.manifest
    for key in ('run_type', 'train_size', 'effective_batch_size', 'seed',
                'train_split_hash', 'val_split_hash', 'pretraining_policy',
                'crop_size', 'validation_mode'):
        left, right = bm.get(key), cm.get(key)
        if left is None or right is None:
            unknown.append(key)
        elif left != right:
            mismatches.append(f'{key}: {left!r} != {right!r}')
    bb, cb = completed_budget(baseline), completed_budget(candidate)
    for key in ('samples', 'updates'):
        if bb[key] is None or cb[key] is None:
            unknown.append(f'completed_{key}')
        elif bb[key] != cb[key]:
            mismatches.append(f'completed_{key}: {bb[key]} != {cb[key]}')
    if baseline.directory == candidate.directory:
        mismatches.append('baseline and candidate are the same directory')
    if bm.get('run_type') == 'smoke' or cm.get('run_type') == 'smoke':
        mismatches.append('smoke runs are not performance baselines')
    return mismatches, sorted(set(unknown)), bb, cb


def delta(left, right):
    return right - left if left is not None and right is not None else None


def make_report(baseline, candidate, bs, cs):
    bi, ci = run_identity(baseline), run_identity(candidate)
    mismatches, unknown, bb, cb = fairness_checks(baseline, candidate)
    rep_delta = delta(bs['representative_miou'], cs['representative_miou'])
    final_delta = delta(bs['final_miou'], cs['final_miou'])
    auc_delta = delta(bs['normalized_auc'], cs['normalized_auc'])
    best_delta = delta(bs['best_miou'], cs['best_miou'])
    evidence = [rep_delta, final_delta, auc_delta]
    finished = (baseline.summary.get('finished') is True and
                candidate.summary.get('finished') is True)
    clean = not baseline.warnings and not candidate.warnings
    enough = bs['window_size'] >= 3 and cs['window_size'] >= 3
    if mismatches:
        verdict = '实验条件或训练预算不一致，不能给出公平提升结论'
    elif not finished or not clean or not enough:
        verdict = '完整、干净的验证证据不足，当前比较只能作为阶段性参考'
    elif all(value is not None and value > 0 for value in evidence):
        verdict = '代表值、最终值和训练过程AUC均提高，存在持续提升迹象，建议换种子或验证折复验'
    elif all(value is not None and value <= 0 for value in evidence):
        verdict = '三项主要证据均未提高，当前未观察到有效改进'
    else:
        verdict = '主要证据方向不一致，改进尚不具有说服力'

    lines = [
        f'## 实验对比：{bi["run_id"]} → {ci["run_id"]}', '',
        f'- 对比时间：`{report_time()}`',
        f'- 基线目录：`{baseline.directory}`',
        f'- 候选目录：`{candidate.directory}`',
        f'- 基线配置：`{bi["config"]}`',
        f'- 候选配置：`{ci["config"]}`',
        f'- 基线模型：`{bi["model"] or "N/A"}`',
        f'- 候选模型：`{ci["model"] or "N/A"}`', '',
        '| 指标 | 基线 | 候选 | 差值（百分点） |',
        '|---|---:|---:|---:|',
        f'| 稳定窗口代表 mIoU | {fmt(bs["representative_miou"])} | '
        f'{fmt(cs["representative_miou"])} | {fmt(rep_delta)} |',
        f'| 稳定窗口均值 | {fmt(bs["tail_mean"])} | '
        f'{fmt(cs["tail_mean"])} | {fmt(delta(bs["tail_mean"], cs["tail_mean"]))} |',
        f'| 稳定窗口标准差 | {fmt(bs["tail_std"])} | '
        f'{fmt(cs["tail_std"])} | {fmt(delta(bs["tail_std"], cs["tail_std"]))} |',
        f'| 最终 mIoU | {fmt(bs["final_miou"])} | {fmt(cs["final_miou"])} | '
        f'{fmt(final_delta)} |',
        f'| 归一化 mIoU AUC | {fmt(bs["normalized_auc"])} | '
        f'{fmt(cs["normalized_auc"])} | {fmt(auc_delta)} |',
        f'| 最佳 mIoU（辅助） | {fmt(bs["best_miou"])} | '
        f'{fmt(cs["best_miou"])} | {fmt(best_delta)} |',
        f'| 完成样本曝光量 | {bb["samples"] or "N/A"} | '
        f'{cb["samples"] or "N/A"} | - |',
        f'| 完成优化器更新 | {bb["updates"] or "N/A"} | '
        f'{cb["updates"] or "N/A"} | - |', '',
        '### 类别对比（稳定窗口中位数）', '',
        '| 类别 | 基线IoU | 候选IoU | 差值 |', '|---|---:|---:|---:|',
    ]
    for name in CLASSES:
        left = bs['class_metrics'][name]['representative']
        right = cs['class_metrics'][name]['representative']
        lines.append(f'| {name} | {fmt(left)} | {fmt(right)} | '
                     f'{fmt(delta(left, right))} |')
    lines.extend(['', '### 条件检查与结论', '', f'- 结论：{verdict}。'])
    if mismatches:
        lines.append('- 不一致项：' + '；'.join(mismatches) + '。')
    if unknown:
        lines.append('- 缺少记录、需人工确认：' + '、'.join(unknown) + '。')
    for label, run in (('基线', baseline), ('候选', candidate)):
        if run.warnings:
            lines.append(f'- {label}日志警告：' + '；'.join(run.warnings) + '。')
    lines.append('- 这是固定本地验证集上的单次实验启发式判断，不是统计显著性或官方测试分数。')
    return '\n'.join(lines)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--baseline', required=True)
    parser.add_argument('--candidate', required=True)
    args = parser.parse_args()
    try:
        baseline, candidate = load_run(args.baseline), load_run(args.candidate)
        report = make_report(baseline, candidate, summarize(baseline),
                             summarize(candidate))
        output = append_result(report)
        print(report)
        print(f'\nAPPENDED: {output}')
    except (OSError, ValueError) as exc:
        parser.exit(1, f'ERROR: {exc}\n')


if __name__ == '__main__':
    main()
