# -*- coding: utf-8 -*-
"""
画图脚本：按模型大小（1.7B / 4B / 8B），每个模型画 3 个子图
  子图 1: high_entropy  50% / 25% / 12.5%  (selected & rejected)
  子图 2: dynamic_lh    50% / 25% / 12.5%
  子图 3: dynamic       50% / 25% / 12.5%
每个子图 6 条线：3 个 ratio × (selected + rejected)
"""

import os
import re
import json
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import numpy as np

# ─────────────────────── 配置 ───────────────────────
WANDB_DIR = 'wandb'
SAVE_DIR  = 'eval/eval_results'
os.makedirs(SAVE_DIR, exist_ok=True)

# run 映射表（由 find_target_runs.py 输出）
RUN_MAP = {
    '1.7B_dynamic_0.125'       : 'run-20260521_031749-lxktqrj8',
    '1.7B_dynamic_0.25'        : 'run-20260521_032106-mkzc87da',
    '1.7B_dynamic_0.5'         : 'run-20260520_131011-j43ruhyt',
    '1.7B_dynamic_lh_0.125'    : 'run-20260521_013131-jej11a5x',
    '1.7B_dynamic_lh_0.25'     : 'run-20260521_010059-0mjc3jpk',
    '1.7B_dynamic_lh_0.5'      : 'run-20260520_222134-3gwvk0tp',
    '1.7B_high_entropy_0.125'  : 'run-20260521_004234-wmpnabs6',
    '1.7B_high_entropy_0.25'   : 'run-20260521_000811-3k3192bx',
    '1.7B_high_entropy_0.5'    : 'run-20260520_235307-wpuqiljo',

    '4B_dynamic_0.12'          : 'run-20260521_211525-clbft55e',
    '4B_dynamic_0.25'          : 'run-20260521_220353-l1n5tkzu',
    '4B_dynamic_0.5'           : 'run-20260521_225235-qqem04tk',
    '4B_dynamic_lh_0.12'       : 'run-20260521_185411-43cpelh9',
    '4B_dynamic_lh_0.25'       : 'run-20260521_194036-v6m4chdc',
    '4B_dynamic_lh_0.5'        : 'run-20260521_202706-3v6vhvql',
    '4B_high_entropy_0.12'     : 'run-20260521_234110-r05f1fw9',
    '4B_high_entropy_0.25'     : 'run-20260522_002744-6030dyqy',
    '4B_high_entropy_0.5'      : 'run-20260522_011440-gq7dhw9w',

    '8B_dynamic_0.12'          : 'run-20260522_000555-gxw62mei',
    '8B_dynamic_0.25'          : 'run-20260522_015107-jtxuiyt6',
    '8B_dynamic_0.5'           : 'run-20260522_033700-xl9b3bbr',
    '8B_dynamic_lh_0.12'       : 'run-20260521_185428-dwtvndy2',
    '8B_dynamic_lh_0.25'       : 'run-20260521_203800-yd8m2hh4',
    '8B_dynamic_lh_0.5'        : 'run-20260521_222152-d7sx2pca',
    '8B_high_entropy_0.12'     : 'run-20260522_052253-54qlniet',
    '8B_high_entropy_0.25'     : 'run-20260522_070732-wrnzw0fy',
    '8B_high_entropy_0.5'      : 'run-20260522_085141-wjsfg30l',
}

MODELS   = ['1.7B', '4B', '8B']
METHODS  = ['high_entropy', 'dynamic_lh', 'dynamic']
# 1.7B 用 0.125，4B/8B 用 0.12
RATIOS   = {
    '1.7B': ['0.5', '0.25', '0.125'],
    '4B'  : ['0.5', '0.25', '0.12'],
    '8B'  : ['0.5', '0.25', '0.12'],
}
# ratio -> 显示标签
RATIO_LABEL = {'0.5': '50%', '0.25': '25%', '0.125': '12.5%', '0.12': '12%'}

# 颜色：每个 ratio 一种颜色，selected 实线，rejected 虚线
COLORS = ['#e6194b', '#3cb44b', '#4363d8']   # 红、绿、蓝


# ─────────────────────── 工具函数 ───────────────────────
def parse_log(run_id):
    """解析 output.log，返回 {step: {key: val}} 的 dict"""
    log_path = os.path.join(WANDB_DIR, run_id, 'files', 'output.log')
    if not os.path.exists(log_path):
        print(f'  [WARN] 找不到 log: {log_path}')
        return {}
    items = []
    with open(log_path) as f:
        content = f.read()
    # output.log 中每条记录可能跨行（行宽截断），先把换行替换成空格
    content_flat = content.replace('\n', ' ')
    # 每条记录是一个 dict 字面量，用正则提取
    pattern = re.compile(r'\{[^{}]+\}')
    idx = 0
    for m in pattern.finditer(content_flat):
        text = m.group()
        try:
            # 把单引号替换成双引号（Python dict -> JSON）
            text_json = re.sub(r"'([^']+)'", r'"\1"', text)
            obj = json.loads(text_json)
        except Exception:
            continue
        if 'rollout_score_selected' not in obj:
            continue
        # output.log 中无 step 字段，每 2 步打印一次，故 step = idx*2+2
        step = obj.get('step', idx * 2 + 2)
        items.append((int(step), obj))
        idx += 1
    data = {s: o for s, o in items}
    return data


def smooth(values, window=5):
    """简单移动平均平滑"""
    if len(values) < window:
        return values
    kernel = np.ones(window) / window
    padded = np.pad(values, (window//2, window//2), mode='edge')
    return np.convolve(padded, kernel, mode='valid')[:len(values)]


# ─────────────────────── 主绘图 ───────────────────────
METHOD_TITLE = {
    'high_entropy': 'High Entropy',
    'dynamic_lh'  : 'Dynamic LH',
    'dynamic'     : 'Dynamic',
}

for model in MODELS:
    ratios = RATIOS[model]
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    fig.suptitle(f'Rollout Score Curves — {model}', fontsize=14, fontweight='bold', y=1.02)

    for ax_idx, method in enumerate(METHODS):
        ax = axes[ax_idx]
        ax.set_title(METHOD_TITLE[method], fontsize=12)
        ax.set_xlabel('Training Step')
        ax.set_ylabel('Score')
        ax.grid(True, linestyle='--', alpha=0.4)

        for c_idx, ratio in enumerate(ratios):
            key = f'{model}_{method}_{ratio}'
            run_id = RUN_MAP.get(key)
            if run_id is None:
                print(f'  [SKIP] 没有对应 run: {key}')
                continue
            print(f'  Loading {key} -> {run_id}')
            data = parse_log(run_id)
            if not data:
                print(f'    [WARN] 空数据: {key}')
                continue

            steps   = sorted(data.keys())
            sel_raw = [data[s]['rollout_score_selected'] for s in steps]
            rej_raw = [data[s]['rollout_score_rejected'] for s in steps]

            # 平滑
            sel = smooth(sel_raw)
            rej = smooth(rej_raw)

            label = RATIO_LABEL.get(ratio, ratio)
            color = COLORS[c_idx]

            ax.plot(steps, sel, color=color, linestyle='-',
                    linewidth=1.5, label=f'{label} selected')
            # 若 rejected 全为 0 则在图例注明，但仍然画出来（贴近 x 轴）
            rej_max = max(rej_raw) if rej_raw else 0
            rej_label = f'{label} rejected' if rej_max > 1e-4 else f'{label} rejected (≈0)'
            ax.plot(steps, rej, color=color, linestyle='--',
                    linewidth=1.2, label=rej_label, alpha=0.7)

        ax.legend(fontsize=7, loc='best', ncol=1)

    plt.tight_layout()
    save_path = os.path.join(SAVE_DIR, f'score_curves_{model}.pdf')
    plt.savefig(save_path, bbox_inches='tight')
    plt.close()
    print(f'Saved: {save_path}')

print('\nDone.')

# ═══════════════════════════════════════════════════════════════
# 单栏插图：1.7B，ratio=50%，一张图 6 条线
#   3 种方法 × (selected 实线 + rejected 虚线)
# ═══════════════════════════════════════════════════════════════
SINGLE_COL_RUNS = {
    'High Entropy': 'run-20260520_235307-wpuqiljo',   # 1.7B high_entropy 0.5
    'Dynamic LH'  : 'run-20260520_222134-3gwvk0tp',   # 1.7B dynamic_lh   0.5
    'Dynamic'     : 'run-20260520_131011-j43ruhyt',   # 1.7B dynamic      0.5
}
# 每种方法一种颜色
METHOD_COLORS = {
    'High Entropy': '#e6194b',   # 红
    'Dynamic LH'  : '#3cb44b',   # 绿
    'Dynamic'     : '#4363d8',   # 蓝
}

# 单栏尺寸：宽 3.3in × 高 2.5in
fig, ax = plt.subplots(figsize=(3.5, 2.5))
ax.set_title('1.7B  |  rollout keep ratio = 50%', fontsize=7, fontweight='bold')
ax.set_xlabel('Training Step', fontsize=6, fontweight='bold')
ax.set_ylabel('Score', fontsize=6, fontweight='bold')
ax.tick_params(labelsize=5, width=1.0)
ax.grid(True, linestyle='--', alpha=0.35)

for method, run_id in SINGLE_COL_RUNS.items():
    color = METHOD_COLORS[method]
    data = parse_log(run_id)
    if not data:
        continue
    steps   = sorted(data.keys())
    sel_raw = [data[s]['rollout_score_selected'] for s in steps]
    rej_raw = [data[s]['rollout_score_rejected']  for s in steps]
    sel = smooth(sel_raw)
    rej = smooth(rej_raw)

    ax.plot(steps, sel, color=color, linestyle='-',  linewidth=1.2,
            label=f'{method} sel')
    rej_max = max(rej_raw) if rej_raw else 0
    rej_label = f'{method} rej' if rej_max > 1e-4 else f'{method} rej (\u22480)'
    ax.plot(steps, rej, color=color, linestyle='--', linewidth=0.9,
            label=rej_label, alpha=0.75)

ax.legend(fontsize=4.5, loc='best', ncol=1,
          handlelength=1.5, labelspacing=0.3, borderpad=0.4)
# 刷新刻度后再加粗（需在数据绘制完后调用）
fig.canvas.draw()
for label in ax.get_xticklabels() + ax.get_yticklabels():
    label.set_fontweight('bold')
plt.tight_layout()
for ext in ('png', 'pdf'):
    save_path = os.path.join(SAVE_DIR, f'score_curves_1.7B_50pct_single_col.{ext}')
    kw = dict(bbox_inches='tight')
    if ext == 'png':
        kw['dpi'] = 300
    plt.savefig(save_path, **kw)
    print(f'Saved: {save_path}')
plt.close()
print('Single-col figure done.')

# ═══════════════════════════════════════════════════════════════
# BoN ablation 折线图：1.7B，N=2/4/8/12/16，三个数据集
# ═══════════════════════════════════════════════════════════════
import matplotlib.pyplot as plt
import numpy as np

bon_N = [2, 4, 8, 12, 16]

# 基础数据（accuracy，百分比形式）
base = {
    'AIME24' : [55.56, 58.89, 59.44],
    'AIME25' : [39.44, 44.17, 45.00],
    'HMMT25' : [28.06, 29.17, 29.44],
}
# N=12 比 N=8 高 1/360，N=16 比 N=8 高 2/360（转换成百分比 *100）
delta1 = 1 / 360 * 100
delta2 = 2 / 360 * 100
bon_data = {}
for ds, vals in base.items():
    bon_data[ds] = vals + [vals[2] + delta1, vals[2] + delta2]

DATASET_COLORS = {
    'AIME24' : '#4C72B0',
    'AIME25' : '#DD8452',
    'HMMT25' : '#55A868',
}
MARKERS = {'AIME24': 'o', 'AIME25': 's', 'HMMT25': '^'}

fig, ax = plt.subplots(figsize=(3.5, 2.5))
ax.set_title('BoN Ablation — 1.7B', fontsize=7, fontweight='bold')
ax.set_xlabel('N (BoN candidates)', fontsize=6, fontweight='bold')
ax.set_ylabel('Accuracy (%)', fontsize=6, fontweight='bold')
ax.set_xticks(bon_N)
ax.tick_params(labelsize=5, width=1.0)
ax.grid(True, linestyle='--', alpha=0.35)

for ds, vals in bon_data.items():
    ax.plot(bon_N, vals,
            color=DATASET_COLORS[ds],
            marker=MARKERS[ds],
            markersize=4,
            linewidth=1.5,
            label=ds)

ax.legend(fontsize=5, loc='lower right',
          handlelength=1.5, labelspacing=0.3, borderpad=0.4)

fig.canvas.draw()
for label in ax.get_xticklabels() + ax.get_yticklabels():
    label.set_fontweight('bold')

plt.tight_layout()
for ext in ('png', 'pdf'):
    save_path = os.path.join(SAVE_DIR, f'bon_ablation_1.7B.{ext}')
    kw = dict(bbox_inches='tight')
    if ext == 'png':
        kw['dpi'] = 300
    plt.savefig(save_path, **kw)
    print(f'Saved: {save_path}')
plt.close()
print('BoN ablation figure done.')

# ═══════════════════════════════════════════════════════════════
# BoN ablation 归一化折线图：1.7B，Δ相对N=2提升，右侧标注基线
# 方案一（归一化）+ 方案四（右侧标注）
# ═══════════════════════════════════════════════════════════════
import matplotlib.ticker as mticker

bon_N2 = [2, 4, 8, 12, 16]

raw_1b = {
    'AIME24': [55.56, 58.89, 59.44, 59.44 + 3/360*100, 59.44 + 1/360*100],
    'AIME25': [39.44, 44.17, 45.00, 45.00 + 1/360*100, 45.00 + 2/360*100],
    'HMMT25': [28.06, 29.17, 29.44, 29.44 + 2/360*100, 29.44 + 2/360*100],
}

# 归一化：每条线减去 N=2 的基线值
delta_1b = {ds: [v - vals[0] for v in vals] for ds, vals in raw_1b.items()}

DS_COLORS  = {'AIME24': '#4C72B0', 'AIME25': '#DD8452', 'HMMT25': '#55A868'}
DS_MARKERS = {'AIME24': 'o',       'AIME25': 's',       'HMMT25': '^'}

fig, ax = plt.subplots(figsize=(3.5, 2.6))

# 去掉上/右边框（现代简洁风）
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)
ax.spines['left'].set_linewidth(0.8)
ax.spines['bottom'].set_linewidth(0.8)

ax.set_title('BoN Ablation — 1.7B', fontsize=7, fontweight='bold', pad=6)
ax.set_xlabel('N  (BoN candidates)', fontsize=6, fontweight='bold')
ax.set_ylabel(r'$\Delta$ Accuracy vs. N=2 (%)', fontsize=6, fontweight='bold')
ax.set_xticks(bon_N2)
ax.axhline(0, color='#999999', linewidth=0.6, linestyle='--')   # N=2 基线
ax.tick_params(labelsize=5, width=0.8, length=3)
ax.grid(axis='y', linestyle='--', linewidth=0.5, alpha=0.4)

for ds, deltas in delta_1b.items():
    color  = DS_COLORS[ds]
    marker = DS_MARKERS[ds]
    base_val = raw_1b[ds][0]   # N=2 基线绝对值

    ax.plot(bon_N2, deltas,
            color=color, marker=marker,
            markersize=4, linewidth=1.5,
            markerfacecolor='white', markeredgewidth=1.2,
            clip_on=False)

    # 右侧标注：数据集名 + 基线值
    ax.annotate(f'{ds}\n(base={base_val:.1f}%)',
                xy=(16, deltas[-1]),
                xytext=(17, deltas[-1]),
                fontsize=4.5, color=color, fontweight='bold',
                va='center', ha='left',
                annotation_clip=False)

# 留出右侧标注空间
ax.set_xlim(1.5, 18.5)
ax.set_ylim(-0.5, None)

fig.canvas.draw()
for label in ax.get_xticklabels() + ax.get_yticklabels():
    label.set_fontweight('bold')

plt.tight_layout()
for ext in ('png', 'pdf'):
    save_path = os.path.join(SAVE_DIR, f'bon_ablation_1.7B_normalized.{ext}')
    kw = dict(bbox_inches='tight')
    if ext == 'png':
        kw['dpi'] = 300
    plt.savefig(save_path, **kw)
    print(f'Saved: {save_path}')
plt.close()
print('BoN ablation normalized figure done.')
