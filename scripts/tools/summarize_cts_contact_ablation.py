#!/usr/bin/env python3
"""Outcome and common-window impact comparison for CTS contact ablations."""
import argparse
import hashlib
import json
from pathlib import Path

import numpy as np

LABELS = {'oracle': '真实触地力', 'estimate_current': '现有估计',
          'estimate_aligned': '对齐估计', 'off': '关闭修正'}
ENGLISH = {'oracle': 'Oracle', 'estimate_current': 'Current estimate',
           'estimate_aligned': 'Aligned estimate', 'off': 'Landing off'}


def write_json(path, value):
    path.write_text(json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False) + '\n')


def summarize(folder):
    results = json.loads((folder / 'results.json').read_text())
    protocol = json.loads((folder / 'protocol.json').read_text())
    if not results['completed']:
        raise ValueError('Experiment is not complete')
    roles = protocol['roles']
    cases = {(case['scene'], case['seed'], case['role']): case for case in results['cases']}
    arrays, common_windows = {}, {}
    for scene in protocol['scenes']:
        for seed in protocol['seeds']:
            stops = []
            for role in roles:
                case = cases[scene, seed, role]
                name = f'{scene}_{role}_{seed}'
                with np.load(folder / (name + '.npz')) as data:
                    arrays[scene, seed, role] = {key: data[key] for key in data.files}
                data = arrays[scene, seed, role]
                end = float(data['time_s'][-1])
                if scene != 'flat':
                    x, y = data['wheel_position_m'][:, :, 0], data['wheel_position_m'][:, :, 1]
                    start_x = case['terrain']['terrain_parameters']['start_x'][0]
                    invalid = (((x >= start_x) & (x <= case['terrain']['goal_x']))
                        & (np.abs(y) > case['terrain']['course_width'] / 2)).any(-1)
                    if invalid.any():
                        end = min(end, float(data['time_s'][np.flatnonzero(invalid)[0]]))
                stops.append(end)
            common_windows[scene, seed] = (protocol['command_start_s'], min(stops))
    groups = []
    for scene in protocol['scenes']:
        for role in roles:
            selected = [cases[scene, seed, role] for seed in protocol['seeds']]
            common_peaks, common_impulses, tilt_values = [], [], []
            for seed in protocol['seeds']:
                start, end = common_windows[scene, seed]
                events = json.loads((folder / f'{scene}_{role}_{seed}.events.json').read_text())
                usable = [event for event in events if event['time_s'] >= start
                          and event['time_s'] + event['integrated_duration_s'] <= end + 1e-9]
                common_peaks.extend(event['peak_normal_force_N'] for event in usable)
                common_impulses.extend(event['normal_impulse_N_s'] for event in usable)
                data = arrays[scene, seed, role]
                mask = (data['time_s'] >= start) & (data['time_s'] <= end)
                tilt_values.extend(data['roll_pitch_rad'][mask].reshape(-1).tolist())
            successful_times = [case['simulated_s'] for case in selected if case['success']]
            count_samples = sum(case['physics_steps'] * 4 for case in selected)
            mistakes = sum((case['false_support_wheel_samples'] or 0) + (case['missed_support_wheel_samples'] or 0) for case in selected)
            groups.append(dict(scene=scene, role=role, n=len(selected),
                success=sum(case['success'] for case in selected), falls=sum(case['fell'] for case in selected),
                bypass=sum(case['bypass'] for case in selected),
                mean_progress_m=float(np.mean([case['progress_m'] for case in selected])),
                mean_success_time_s=float(np.mean(successful_times)) if successful_times else None,
                common_window_roll_pitch_rms_deg=float(np.sqrt(np.mean(np.square(tilt_values))) * 180 / np.pi) if tilt_values else None,
                common_window_event_peak_p95_N=float(np.quantile(common_peaks, .95)) if common_peaks else None,
                common_window_impulse_p95_Ns=float(np.quantile(common_impulses, .95)) if common_impulses else None,
                common_window_event_count=len(common_peaks),
                modified_wheel_samples=sum(case['modified_wheel_samples'] for case in selected),
                modified_fraction=sum(case['modified_wheel_samples'] for case in selected) / count_samples,
                minimum_kp_scale=min(case['min_kp_scale'] for case in selected),
                max_joint_correction_rad=max(case['max_joint_correction_rad'] for case in selected),
                contact_fsm_agreement=1 - mistakes / count_samples if role != 'off' else None))
    pairs = []
    for scene in protocol['scenes']:
        for seed in protocol['seeds']:
            oracle = cases[scene, seed, 'oracle']
            off = cases[scene, seed, 'off']
            pairs.append(dict(scene=scene, seed=seed, oracle_success=oracle['success'], off_success=off['success'],
                oracle_fell=oracle['fell'], off_fell=off['fell'],
                off_progress_difference_percent=100 * (off['progress_m'] / oracle['progress_m'] - 1),
                common_measurement_window_s=list(common_windows[scene, seed])))
    aggregate = dict(groups=groups, oracle_off_pairs=pairs,
        completed_cases=len(results['cases']), total_physics_steps=results['steps'],
        hardware_commands=0, optimizer_updates=0, source_model_sha256=results['model_sha256'])
    write_json(folder / 'aggregate.json', aggregate)

    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    colors = ['#222222', '#D55E00', '#0072B2', '#009E73']
    scenes = list(protocol['scenes'])
    fig, axes = plt.subplots(1, 3, figsize=(12, 4.2), constrained_layout=True)
    for index, scene in enumerate(scenes):
        ax = axes[index]
        for i, role in enumerate(roles):
            selected = [cases[scene, seed, role] for seed in protocol['seeds']]
            for j, case in enumerate(selected):
                symbol = 'o' if case['success'] else ('x' if case['fell'] else '^')
                ax.scatter(i + (j-1) * .12, case['progress_m'], color=colors[i], marker=symbol, s=50)
        ax.set_xticks(range(len(roles)), [ENGLISH[r] for r in roles], rotation=25, ha='right', fontsize=8)
        ax.set_ylabel('Forward root progress (m)')
        ax.set_title(scene + ': circle=success, x=fall, triangle=incomplete', fontsize=9)
        ax.grid(axis='y', alpha=.25)
    fig.savefig(folder / 'motion_comparison.png', dpi=160)
    plt.close(fig)
    fig, axes = plt.subplots(1, 2, figsize=(10, 4), constrained_layout=True)
    width = .18
    for i, role in enumerate(roles):
        selected = [next(group for group in groups if group['scene'] == scene and group['role'] == role) for scene in scenes]
        x = np.arange(len(scenes)) + (i-1.5)*width
        axes[0].bar(x, [g['common_window_event_peak_p95_N'] or 0 for g in selected], width=width, color=colors[i], label=ENGLISH[role])
        axes[1].bar(x, [g['common_window_roll_pitch_rms_deg'] or 0 for g in selected], width=width, color=colors[i])
    for ax in axes:
        ax.set_xticks(range(len(scenes)), scenes)
        ax.grid(axis='y', alpha=.2)
    axes[0].set_ylabel('Contact event peak p95 (N)')
    axes[1].set_ylabel('Roll/pitch RMS (degrees)')
    axes[0].set_title('Same-seed common time window before first lane exit/end')
    axes[1].set_title('Same common window; excludes start-up')
    fig.legend(*axes[0].get_legend_handles_labels(), loc='outside lower center', ncols=4, fontsize=8)
    fig.savefig(folder / 'impact_posture_comparison.png', dpi=160)
    plt.close(fig)

    lines = ['# CTS10000：触地反馈与落地修正闭环消融', '',
        '本轮回答：关闭落地目标修正和 PD 调整后，CTS 是否仍能运动，以及通过能力与冲击如何变化。', '',
        f'正式完成 {len(results["cases"])} 个独立闭环回合、{results["steps"]} 个物理步。网络使用实际部署 CTS10000 ONNX（含共享 decoder），全部回合未改权重、未训练、未连接真机。', '',
        '## 结果判断', '']
    off_basic = [g for g in groups if g['role'] == 'off' and g['scene'] in ('flat', 'descent')]
    if off_basic and all(g['success'] == g['n'] for g in off_basic):
        lines += ['关闭落地修正后，本轮平地和下楼均全部完成，说明 CTS10000 保留独立基础运动能力。触地估计不是网络产生运动动作的必要输入。', '']
    up_oracle = next((g for g in groups if g['scene'] == 'ascent' and g['role'] == 'oracle'), None)
    up_off = next((g for g in groups if g['scene'] == 'ascent' and g['role'] == 'off'), None)
    if up_oracle and up_off:
        lines += [f'本轮上楼：真实触地力配合完整修正完成 {up_oracle["success"]}/{up_oracle["n"]}，关闭修正完成 {up_off["success"]}/{up_off["n"]}。现有落地控制没有展示稳定提高上楼通过能力；各组目前都未达到稳定上楼。小样本不能据此认定关闭控制普遍更好。', '']
    lines += ['控制开关会改变部分种子的运动结果，因此不能一概说只影响声音；冲击峰值与冲量也不是同向变化，没有证据声称当前修正必然更安静。', '',
        '## 运动结果', '',
        '| 场景 | 触地/控制方式 | 完成 | 摔倒 | 曾出楼梯边界 | 成功用时均值 s |',
        '|---|---|---:|---:|---:|---:|']
    for group in groups:
        duration = f'{group["mean_success_time_s"]:.2f}' if group['mean_success_time_s'] is not None else '—'
        lines.append(f'| {group["scene"]} | {LABELS[group["role"]]} | {group["success"]}/{group["n"]} | {group["falls"]}/{group["n"]} | {group["bypass"]}/{group["n"]} | {duration} |')
    lines += ['', '平地“完成”指无摔倒且前进至少 3 m；楼梯完成要求所有轮子在路线内越过终点、真实接触最终平台并保持至少 0.25 s，发生过绕出楼梯边界不算成功。',
        '前进距离大不能替代楼梯通过：运动图中三角形表示未完成，可能已经绕到楼梯外的地面继续行驶。', '',
        '## 姿态与落地冲击', '',
        '每个种子采用相同起始时刻和结束时刻：1 s 零命令站稳后开始，截止四种方式中最早结束或首次越出楼梯边界。以下只比较该共同时间窗内完整落地事件，避免摔倒早停/绕到平地造成“更安静”的假象。姿态按时间样本池化，冲击按完整事件池化，并非三个种子的等权均值。', '',
        '| 场景 | 方式 | 姿态 RMS ° | 事件峰值 p95 N | 冲量 p95 N·s | 修正轮样本数（全程） |',
        '|---|---|---:|---:|---:|---:|']
    for group in groups:
        numbers = [group[k] for k in ('common_window_roll_pitch_rms_deg', 'common_window_event_peak_p95_N', 'common_window_impulse_p95_Ns')]
        values = ['—' if v is None else f'{v:.2f}' for v in numbers]
        lines.append(f'| {group["scene"]} | {LABELS[group["role"]]} | {values[0]} | {values[1]} | {values[2]} | {group["modified_wheel_samples"]} |')
    lines += ['', '这些是仿真机械冲击代理，不是声音录音或声压 dB。冲击代理变化不自动等于声音变化。', '',
        '## 接线与实验范围', '',
        '- 使用现有 deploy_mujoco.M20Contract、StudentPerception、StudentMapAdapter、MujocoLanding 和 LandingController；只新增实验入口。ONNX 输入为 policy789 + 原始 command3，输出 nominal16。动作历史保留 nominal，保存感知噪声/地图延迟/遮挡与动作延迟未修改。',
        '- oracle 使用上一完整 solve 的逐轮净接触力 world +Z，仅统计轮子与地形的接触，采用 Isaac 落地输入的竖直分量定义。没有把本步未来接触力供给本步控制。所有反馈在下一控制步使用，源年龄 5 ms。训练关闭 self-collision；本轮沿用原 MuJoCo 模型，未额外记录排除的轮-机身自碰接触，不能声称所有碰撞集合与 PhysX 完全一致。',
        '- estimate_current 调用现有 Diagnostics.consume；estimate_aligned 以一个物理区间末端速度差分回推该区间源状态的力，两者只使用可传感的 q/dq/torque/姿态/比力/角速度，不读取 oracle 力或解析关节加速度作为估计输入。',
        '- 估计回合显式假设数值有限且 Jacobian 可解的估计有效，只是离线闭环能力消融；没有把生产 contact_valid 改成 true。off 直接调用原 nominal PD，修正量和增益变化均为零。',
        '- 四组固定同一模型、初态、命令、原生学生噪声与每个种子；1 s 零命令后前进命令 0.5 m/s。楼梯 10 级、高 0.15 m、深 0.30 m；原生上楼宽 1 m、起点 x=0.25 m，下楼宽 4 m、起点 x=0.5 m。每个场景内部四组完全同宽，不能据此直接比较上下楼难度或外推窄梯下楼。每组 3 个种子，结果是小样本对照，不代表总体成功率估计。',
        '- 原模型参数、轮柱体碰撞、关节/执行器限制与轮 armature 保持部署基准；没有为任一方式软化接触或调控制增益。MuJoCo 与训练 PhysX/真实电机仍有差异。',
        f'- 单进程单线程 CPU，实际峰值 RSS {results["max_rss_kib"]/1024:.1f} MiB，正式运行墙钟 {results["elapsed_wall_s"]:.1f} s。Torch 共享库虚拟映射约 4.4 GiB，因此地址空间上限 12 GiB，预估 RSS <1 GiB。',
        '- 冻结协议和全部来源 SHA 见 protocol.json；逐回合 JSON/NPZ/events 留存，aggregate.json 含共同窗口与同种子成对结果。', '',
        '## 独立复核', '',
        '同 Codex family reviewer 独立重算全部 NPZ 的运动/姿态/接触状态指标，以及 12 组共同时间窗冲击聚合，结果逐数一致；独立短复跑验证实际 PD 被修改、Euler 差分源时刻正确、原始命令与 nominal 历史接线正确。边界不完整事件未纳入冲击聚合。复核 JSON 在 independent_review/；缺异构 family。', '',
        '## 复现', '', '```bash',
        'OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 \\',
        '/home/ubuntu/miniconda3/envs/m20_wzh/bin/python \\',
        '  scripts/tools/evaluate_cts_contact_ablation.py \\',
        f'  --protocol {folder}/protocol.json --output /tmp/cts_contact_ablation_new', '```', '']
    (folder / 'REPORT.md').write_text('\n'.join(lines))
    print(json.dumps(dict(report=str(folder / 'REPORT.md'), completed_cases=len(results['cases']),
                          physics_steps=results['steps']), ensure_ascii=False))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('folder', type=Path)
    summarize(parser.parse_args().folder.resolve())
