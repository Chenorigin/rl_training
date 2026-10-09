#!/usr/bin/env python3
"""Export full TensorBoard scalar summaries and selected diagnostic plots."""
import argparse
import json
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
from tensorboard.backend.event_processing.event_accumulator import EventAccumulator


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('run', type=Path)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    accumulator = EventAccumulator(str(args.run), size_guidance={'scalars': 0})
    accumulator.Reload()
    required = ['Train/mean_reward', 'Episode_Reward/stair_front_wheel_lift',
                'Curriculum/terrain_level_pyramid_stairs_inv']
    for tag in required:
        if tag not in accumulator.Tags()['scalars']:
            raise ValueError(f'Missing required diagnostic scalar: {tag}')
    summary, series = {}, {}
    for tag in accumulator.Tags()['scalars']:
        events = accumulator.Scalars(tag)
        steps = np.array([e.step for e in events])
        values = np.array([e.value for e in events])
        series[tag] = (steps, values)
        n = len(values)
        summary[tag] = dict(first_step=int(steps[0]), last_step=int(steps[-1]),
            samples=n, first_500=float(values[:500].mean()),
            middle_500=float(values[max(0,n//2-250):n//2+250].mean()),
            last_500=float(values[-500:].mean()), min=float(values.min()), max=float(values.max()))
    args.output.mkdir(parents=True, exist_ok=True)
    (args.output/'tensorboard_summary.json').write_text(json.dumps(summary, indent=2)+'\n')
    panels = [
        ('Reward and survival', ['Train/mean_reward'], 'mean episode return'),
        ('Terrain curriculum (center reset)', ['Curriculum/terrain_level_pyramid_stairs',
            'Curriculum/terrain_level_pyramid_stairs_inv', 'Curriculum/terrain_level_overall'], 'mean level'),
        ('Ascent landing events', ['Curriculum/step_completion/up_attempts',
            'Curriculum/step_completion/up_successes', 'Curriculum/step_completion/up_front_violations'], 'events / reset episode'),
        ('Ascent reward contributions', ['Episode_Reward/stair_front_wheel_lift',
            'Episode_Reward/stair_front_alternation', 'Episode_Reward/stair_up_step_completion',
            'Episode_Reward/stair_up_rear_step_completion'], 'weighted episode sum / max duration'),
        ('Large reward contributions', ['Episode_Reward/track_lin_vel_xy_exp',
            'Episode_Reward/track_ang_vel_z_exp', 'Episode_Reward/stair_joint_acceleration_cost',
            'Episode_Reward/action_smooth_l2'], 'weighted episode sum / max duration'),
        ('Descent and visibility', ['Curriculum/step_completion/down_success_rate',
            'Curriculum/step_completion/forward_gate_fraction',
            'Curriculum/step_completion/nearby_gate_fraction'], 'logged fraction'),
    ]
    fig, axes = plt.subplots(3, 2, figsize=(13, 11), layout='constrained')
    for ax, (title, tags, ylabel) in zip(axes.flat, panels):
        for tag in tags:
            if tag not in series:
                raise ValueError(f'Missing plotted scalar: {tag}')
            x, y = series[tag]
            window = min(200, len(y))
            smooth = np.convolve(y, np.ones(window)/window, mode='valid')
            ax.plot(x[window-1:], smooth, label=tag.split('/')[-1], linewidth=1.2)
        ax.set(title=title, xlabel='PPO iteration', ylabel=ylabel)
        ax.grid(alpha=.2)
        ax.legend(fontsize=8)
        if title == 'Ascent reward contributions':
            ax.set_yscale('symlog', linthresh=1e-6)
    fig.suptitle(args.run.name + ' | 200-iteration moving mean')
    fig.savefig(args.output/'tensorboard_diagnosis.png', dpi=160)
    plt.close(fig)
    print(json.dumps({'run': str(args.run), 'scalar_count': len(summary),
                      'last_iteration': summary['Train/mean_reward']['last_step'],
                      'output': str(args.output)}, indent=2))


if __name__ == '__main__':
    main()
