"""One comparison figure from existing measured metrics; no simulation imports."""
from pathlib import Path
import json
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.font_manager import FontProperties
from matplotlib.patches import Patch

OUT = Path(__file__).resolve().parent
data = json.loads((OUT / 'metrics.json').read_text())
font = FontProperties(fname='/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc')
plt.rcParams.update({'font.family': font.get_name(), 'font.size': 12,
                     'axes.unicode_minus': False, 'axes.spines.top': False,
                     'axes.spines.right': False, 'pdf.fonttype': 42,
                     'axes.titleweight': 'bold'})
models = ('baseline', 'quiet_landing_v1', 'student')
colors = ('#929AA5', '#2574AB', '#E38332')
labels = ('Baseline · 214700', 'Quiet Landing V1 · 257100', '学生 · 22800')
sections = ('ascent', 'descent', 'all_terrain')
section_labels = ('上楼', '下楼', '全程')
stats = data['primary_case_mean_statistics']
fig, axes = plt.subplots(2, 3, figsize=(17, 11))
fig.subplots_adjust(left=.07, right=.975, top=.77, bottom=.17, wspace=.27, hspace=.64)
fig.patch.set_facecolor('#F6F8FB')

def number(v):
    return f'{v:.1f}' if v >= 100 else f'{v:.2f}' if v >= 1 else f'{v:.3f}'

def clean_panel(ax, key, title, unit, annotate=False):
    x, width = np.arange(3), .23
    peak = 0
    for i, model in enumerate(models):
        values = [stats[model]['sections'][section][key]['mean'] for section in sections]
        peak = max(peak, max(values))
        bars = ax.bar(x + (i-1)*width, values, width, color=colors[i], zorder=3)
        ax.bar_label(bars, labels=[number(v) for v in values], padding=4+10*(i % 2), fontsize=9)
    ax.set_xticks(x, section_labels)
    ax.set_title(title, loc='left', fontsize=14, pad=13)
    ax.set_ylabel(unit, fontsize=11)
    ax.set_ylim(0, peak*1.27)
    ax.grid(axis='y', color='#DDE3EA', alpha=.85, zorder=0)
    if annotate:
        changes=[]
        for sec in sections:
            a=stats['quiet_landing_v1']['sections'][sec][key]['mean']
            b=stats['student']['sections'][sec][key]['mean']
            changes.append(100*(b/a-1))
        ax.text(.5, -.25, '学生相对 V1：'+' / '.join(f'{v:+.1f}%' for v in changes),
                transform=ax.transAxes, ha='center', fontsize=11, color='#815125')

clean_panel(axes[0,0], 'speed_p95_m_s', 'A  触地前速度 P95', 'm/s', True)
clean_panel(axes[0,1], 'energy_mean_J_kg', 'B  单次触地比动能均值', 'J/kg', True)
clean_panel(axes[0,2], 'force_peak_mean_N', 'C  触地后 40ms 力峰均值', 'N', True)
clean_panel(axes[1,0], 'energy_per_xy_m', 'D  每米累计触地比动能', 'J/kg/m', True)

def noise_panel(ax,key,title,unit):
    profiles=('clean','train','realistic')
    values=[data['runs']['student_'+p+'_center']['sections']['descent'][key] for p in profiles]
    bars=ax.bar(range(3),values,color=('#2574AB','#3B9F91','#E38332'),width=.6,zorder=3)
    ax.bar_label(bars,labels=[number(v) for v in values],padding=5,fontsize=11)
    ax.set_xticks(range(3),['干净观测','训练噪声','realistic'])
    ax.set_title(title,loc='left',fontsize=14,pad=13)
    ax.set_ylabel(unit,fontsize=11)
    ax.set_ylim(0,max(values)*1.25)
    ax.grid(axis='y',color='#DDE3EA',alpha=.85,zorder=0)
    ax.text(.5,-.25,f'realistic 相对 clean：+{100*(values[2]/values[0]-1):.1f}%',
            transform=ax.transAxes,ha='center',fontsize=11,color='#A8581D')

noise_panel(axes[1,1],'energy_mean_J_kg','E  学生下楼：单次比动能','J/kg')
noise_panel(axes[1,2],'force_peak_p95_N','F  学生下楼：40ms 力峰 P95','N')
for ax in axes.flat:
    ax.set_facecolor('#FFFFFF')
    ax.tick_params(labelsize=11)

fig.text(.07,.965,'学生模型的轻盈落地能力：保留与回退',fontsize=23,weight='bold',color='#253449',va='top')
fig.text(.07,.917,'三模型 clean 对照：相同 MuJoCo 模型、PD、地形与路线反馈；三个初态均完成且未跌倒',
         fontsize=13,color='#59677B',va='top')
fig.legend(handles=[Patch(facecolor=color,label=label) for color,label in zip(colors,labels)],
           loc='upper left',bbox_to_anchor=(.066,.887),ncol=3,frameon=False,fontsize=12)
fig.text(.07,.817,'A–D：三个初态回合等权均值',fontsize=12,color='#59677B')
fig.text(.383,.447,'E–F：同一学生、同一中心初态，每种噪声仅一回合',fontsize=12,color='#59677B')
fig.text(.07,.07,'主要变化：上楼速度 P95 +7.1%；全程每米触地比动能 +16.0%（学生相对 V1）。',
         fontsize=13,weight='bold',color='#253449')
fig.text(.07,.035,'realistic 含感知噪声和 20ms 动作延迟；噪声图不用于三模型同噪声排名。比动能是冲击代理量，不是声音或分贝。',
         fontsize=11,color='#59677B')
for suffix in ('png','pdf'):
    fig.savefig(OUT/f'comparison_overview.{suffix}',dpi=190,facecolor=fig.get_facecolor())
plt.close(fig)
print(OUT/'comparison_overview.png')
