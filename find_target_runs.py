import os
import re

wandb_dir = 'wandb'
target_methods = ['dynamic_lh', 'high_entropy', 'dynamic']
target_ratios = ['0.125', '0.12', '0.25', '0.5']  # 1.7B用0.125，4B/8B用0.12
target_models = ['1.7B', '4B', '8B']

results = {}

for run_id in sorted(os.listdir(wandb_dir)):
    if not run_id.startswith('run-'):
        continue
    log_path = os.path.join(wandb_dir, run_id, 'files', 'output.log')
    cfg_path = os.path.join(wandb_dir, run_id, 'files', 'config.yaml')
    if not os.path.exists(log_path) or not os.path.exists(cfg_path):
        continue
    # 检查是否成功完成
    with open(log_path) as f:
        content = f.read()
    if 'train_runtime' not in content:
        continue
    # 从 config.yaml 提取关键参数
    with open(cfg_path) as f:
        cfg = f.read()
    m_mode  = re.search(r'rollout_select_mode:\s*\n\s*value:\s*(\S+)', cfg)
    m_ratio = re.search(r'rollout_keep_ratio:\s*\n\s*value:\s*(\S+)', cfg)
    m_model = re.search(r'model_name:\s*\n\s*value:\s*(\S+)', cfg)
    if not m_mode or not m_ratio or not m_model:
        continue
    mode       = m_mode.group(1)
    ratio      = m_ratio.group(1)
    model_path = m_model.group(1)
    # 提取模型大小
    model_size = None
    for s in target_models:
        if s in model_path:
            model_size = s
            break
    if model_size is None:
        continue
    if mode not in target_methods:
        continue
    if ratio not in target_ratios:
        continue
    key = f'{model_size}_{mode}_{ratio}'
    # 若同一 key 有多个成功 run，保留最新的（run_id 按时间排序）
    if key not in results or run_id > results[key]:
        results[key] = run_id

print(f"{'Key':<45} {'Run ID'}")
print('-' * 80)
for k in sorted(results):
    print(f"{k:<45} {results[k]}")

print(f'\nFound {len(results)} / 27 target runs')

# 列出缺失的
expected = [f'{m}_{method}_{r}' for m in target_models for method in target_methods for r in target_ratios]
missing = [e for e in expected if e not in results]
if missing:
    print('\nMissing:')
    for m in missing:
        print(f'  {m}')
