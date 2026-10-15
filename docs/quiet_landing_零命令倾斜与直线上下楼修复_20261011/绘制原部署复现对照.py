"""Plot measured interactive deployment replay; no course controller was used."""
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.font_manager import FontProperties
import numpy as np

ROOT = Path(__file__).resolve().parent
TAGS = ("V1_255900", "V1.5_230300", "V1.5_233500")
COLORS = ("#5377a6", "#da8546", "#469a83")
font = FontProperties(fname="/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc")
plt.rcParams.update({"font.family": font.get_name(), "axes.unicode_minus": False,
                     "font.size": 11, "axes.spines.top": False,
                     "axes.spines.right": False})


def yaw_degrees(quat):
    w, x, y, z = quat.T
    return np.rad2deg(np.unwrap(np.arctan2(2*(w*z+x*y), 1-2*(y*y+z*z))))


def main():
    fig, axes = plt.subplots(2, 2, figsize=(13, 9), constrained_layout=True)
    for index, (tag, color) in enumerate(zip(TAGS, COLORS)):
        pose = np.load(ROOT/f"{tag}_零命令_姿态.npz")
        quiet = np.load(ROOT/f"{tag}_零命令_原部署回放_quiet.npz")
        t = pose["time"]; active = t >= 6
        start = np.flatnonzero(active)[0]
        axes[0, 0].plot(t[active]-6, pose["pos"][active, 0]-pose["pos"][start, 0],
                        label=tag, color=color)
        yaw = yaw_degrees(pose["quat"])
        axes[0, 1].plot(t[active]-6, yaw[active]-yaw[start], color=color, label=tag)
        force = quiet["normal_force_N"]
        before = force[(quiet["time"] >= 5.5) & (quiet["time"] < 6)].mean(axis=0)
        after = force[quiet["time"] >= 10].mean(axis=0)
        positions = np.arange(4)+(index-1)*.24
        axes[1, 0].bar(positions, after, width=.23, color=color, label=tag)
        axes[1, 0].plot(positions, before, "_", color="black", markersize=12,
                        label="切入前固定站姿" if index == 0 else None)
        diagonal = (force[:, 0]+force[:, 3]-force[:, 1]-force[:, 2])/np.maximum(force.sum(axis=1), 1e-6)
        # One-second bin means preserve sign; instantaneous contact spikes are not shown as bias.
        bins = np.arange(5, 36)
        values = [100*diagonal[(quiet["time"] >= b) & (quiet["time"] < b+1)].mean() for b in bins]
        axes[1, 1].plot(bins+.5-6, values, color=color, label=tag)
    titles = ("按 c 后前向位移（30 秒零速度命令）", "按 c 后世界航向变化",
              "四轮平均法向力（切入后第 4–30 秒）", "对角组力差：左前＋右后减右前＋左后")
    for ax, title in zip(axes.flat, titles):
        ax.set_title(title); ax.grid(alpha=.2, axis="y")
    axes[0, 0].set(xlabel="策略运行时间（秒）", ylabel="世界前向位移（米）")
    axes[0, 1].set(xlabel="策略运行时间（秒）", ylabel="相对切入时航向（度）")
    axes[1, 0].set(xticks=np.arange(4), xticklabels=["左前 FL", "右前 FR", "左后 HL", "右后 HR"], ylabel="法向力（牛）")
    axes[1, 1].set(xlabel="相对按 c 时间（秒）", ylabel="力差占总支撑力（%，每秒均值）")
    axes[0, 0].legend(); axes[1, 0].legend(fontsize=9)
    fig.suptitle("原部署复现：仅切入策略即出现漂移和对角受力偏置\n三个模型均未摔倒；没有部署端居中、航向或路径纠偏", fontsize=15)
    fig.savefig(ROOT/"零命令按c切入策略_前向漂移_世界航向变化与四轮对角受力偏置对照.png", dpi=180)
    plt.close(fig)

    fig, axes = plt.subplots(2, 1, figsize=(13, 8), constrained_layout=True)
    for tag, color in zip(TAGS, COLORS):
        pose = np.load(ROOT/f"{tag}_仅w_姿态.npz")
        active = (pose["time"] >= 8) & (pose["pos"][:, 0] <= 8)
        pos = pose["pos"][active]; yaw = yaw_degrees(pose["quat"])[active]
        axes[0].plot(pos[:, 0], pos[:, 1], color=color, label=tag)
        axes[1].plot(pos[:, 0], yaw, color=color, label=tag)
    for ax in axes:
        for left, right, label, color in ((1, 2.8, "上楼", "#d4e7f6"), (2.8, 4.8, "平台", "#e7e7e7"), (4.8, 6.6, "下楼", "#f5e4d4")):
            ax.axvspan(left, right, color=color, alpha=.5)
            ax.text((left+right)/2, .98, label, transform=ax.get_xaxis_transform(), ha="center", va="top")
        ax.axhline(0, color="black", linewidth=1, linestyle="--")
        ax.set_xlim(-1, 8); ax.set_xlabel("世界前向位置 x（米）"); ax.grid(alpha=.2)
    axes[0].set(ylabel="世界横向偏移 y（米）", title="进入楼梯、经过平台和下楼的实际路径（目标 y＝0）")
    axes[1].set(ylabel="世界航向（度）", title="同一路段的实际航向（目标沿初始世界 x 方向）")
    axes[0].legend()
    fig.suptitle("原部署仅按 w：机身前向速度 0.7 米/秒、侧向和偏航命令均为零\n上楼与下楼均有偏航；三个模型均未摔倒；没有外部纠偏", fontsize=15)
    fig.savefig(ROOT/"仅按w直行进入楼梯_登顶平台及下楼_世界轨迹与偏航角对照.png", dpi=180)
    plt.close(fig)


if __name__ == "__main__":
    main()
