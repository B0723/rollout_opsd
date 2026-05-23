import wandb
import os
import glob

api = wandb.Api()
entity = "2075379828-nanjing-university-of-aeronautics-and-astrona"
project = "OPSD"

# 读本地所有 run id → 目录名的映射
wandb_dir = os.path.expanduser("~/buyixin02/rollout_opsd/wandb")
local_dirs = glob.glob(os.path.join(wandb_dir, "run-*/"))
local_ids = {}
for d in local_dirs:
    dirname = os.path.basename(d.rstrip("/"))
    run_id = dirname.split("-")[-1]
    local_ids[run_id] = dirname

# 从 WandB API 获取所有 run
runs = api.runs(f"{entity}/{project}")
for run in runs:
    dir_name = local_ids.get(run.id, "not_local")
    print(f"{dir_name:<50}  {run.name}")
