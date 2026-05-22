"""
R3-Affinity Score 离线分析

从已有的 length_analysis.jsonl 读取 seq_entropy 和 kl_loss，
计算四种 rollout 质量分数：
  S_R3      = (1 - H_norm) * K_norm           -- R3-Affinity（确信 × 高KL）
  S_soft_or = H_norm + K_norm - H_norm*K_norm  -- Soft-OR（任一轴有信号）
  S_eff     = K_norm / (1 + lambda * L_norm)   -- Efficiency（KL/长度）
  S_comp    = K_norm*(1-H_norm) + gamma*K_norm*H_norm  -- Composite

输出：
  - 终端打印各分数 Top-50% vs Bottom-50% 正确率对比
  - 各四象限的分数分布
  - score_r3_affinity.jsonl        （每行追加了四个分数字段）
  - score_r3_affinity_summary.json

Usage:
    python score_r3_affinity.py \\
        --input  train_output/length_analysis/length_analysis.jsonl \\
        --output train_output/length_analysis \\
        --top_ratio 0.5 \\
        --lambda_eff 1.0 \\
        --gamma_comp 0.5
"""

import argparse
import json
import os


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _minmax_norm(vals):
    """Batch-level min-max normalize to [0, 1]."""
    vmin, vmax = min(vals), max(vals)
    span = vmax - vmin
    if span < 1e-12:
        return [0.5] * len(vals)
    return [(v - vmin) / span for v in vals]


def _mean(vals):
    vals = [v for v in vals if v is not None]
    return sum(vals) / len(vals) if vals else None


def _fmt(v, fmt="4f"):
    if v is None:
        return "N/A"
    try:
        return f"{v:.{fmt}}"
    except Exception:
        return str(v)


# ---------------------------------------------------------------------------
# Args
# ---------------------------------------------------------------------------

def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--input",  default="train_output/length_analysis/length_analysis.jsonl")
    p.add_argument("--output", default="train_output/length_analysis")
    p.add_argument("--top_ratio",   type=float, default=0.5,
                   help="Top fraction kept as 'high score' group.")
    p.add_argument("--lambda_eff",  type=float, default=1.0,
                   help="Length penalty coefficient for S_eff.")
    p.add_argument("--gamma_comp",  type=float, default=0.5,
                   help="R1 weight for S_comp (0=pure R3, 1=pure KL).")
    return p.parse_args()


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    args = parse_args()
    os.makedirs(args.output, exist_ok=True)
    out_jsonl   = os.path.join(args.output, "score_r3_affinity.jsonl")
    out_summary = os.path.join(args.output, "score_r3_affinity_summary.json")

    # ----------------------------------------------------------------
    # Load records
    # ----------------------------------------------------------------
    records = []
    with open(args.input) as f:
        for line in f:
            line = line.strip()
            if line:
                records.append(json.loads(line))

    print(f"Loaded {len(records)} records from {args.input}")

    # Only keep records with valid kl_loss and seq_entropy
    valid = [r for r in records if r.get("kl_loss") is not None
             and r.get("seq_entropy") is not None]
    invalid_count = len(records) - len(valid)
    if invalid_count:
        print(f"  Warning: {invalid_count} records missing kl_loss/seq_entropy, excluded.")

    # ----------------------------------------------------------------
    # Extract raw values and normalize
    # ----------------------------------------------------------------
    H_raw = [r["seq_entropy"]    for r in valid]
    K_raw = [r["kl_loss"]        for r in valid]
    L_raw = [r["rollout_length"] for r in valid]

    H_norm = _minmax_norm(H_raw)
    K_norm = _minmax_norm(K_raw)
    L_norm = _minmax_norm(L_raw)

    lam = args.lambda_eff
    gam = args.gamma_comp

    # Compute the four scores
    s_r3      = [(1 - h) * k              for h, k, _ in zip(H_norm, K_norm, L_norm)]
    s_soft_or = [h + k - h * k            for h, k, _ in zip(H_norm, K_norm, L_norm)]
    s_eff     = [k / (1 + lam * l)        for _, k, l in zip(H_norm, K_norm, L_norm)]
    s_comp    = [k * (1 - h) + gam * k * h for h, k, _ in zip(H_norm, K_norm, L_norm)]

    all_scores = {
        "S_R3":      s_r3,
        "S_soft_or": s_soft_or,
        "S_eff":     s_eff,
        "S_comp":    s_comp,
    }

    # Attach scores to records
    for i, r in enumerate(valid):
        r["H_norm"]    = H_norm[i]
        r["K_norm"]    = K_norm[i]
        r["S_R3"]      = s_r3[i]
        r["S_soft_or"] = s_soft_or[i]
        r["S_eff"]     = s_eff[i]
        r["S_comp"]    = s_comp[i]

    # ----------------------------------------------------------------
    # Per-score: Top vs Bottom split
    # ----------------------------------------------------------------
    top_k = int(len(valid) * args.top_ratio)
    summary = {
        "total_valid": len(valid),
        "top_ratio": args.top_ratio,
        "lambda_eff": lam,
        "gamma_comp": gam,
        "scores": {},
    }

    def _acc(group):
        ev  = [r for r in group if r.get("correct") is not None]
        cor = sum(1 for r in ev if r["correct"])
        return (cor / len(ev) if ev else None), len(ev), cor

    print(f"\n{'=' * 70}")
    print(f"ROLLOUT SCORE ANALYSIS  top_ratio={args.top_ratio:.0%}  n={len(valid)}")
    print(f"  lambda_eff={lam},  gamma_comp={gam}")
    print(f"{'=' * 70}")

    for score_name, sv in all_scores.items():
        order = sorted(range(len(valid)), key=lambda i: sv[i], reverse=True)
        top_recs    = [valid[i] for i in order[:top_k]]
        bottom_recs = [valid[i] for i in order[top_k:]]

        acc_top,    ev_top,    cor_top    = _acc(top_recs)
        acc_bottom, ev_bottom, cor_bottom = _acc(bottom_recs)
        acc_all,    ev_all,    cor_all    = _acc(valid)

        avg_kl_top     = _mean([r["kl_loss"]       for r in top_recs])
        avg_kl_bottom  = _mean([r["kl_loss"]       for r in bottom_recs])
        avg_H_top      = _mean([r["seq_entropy"]   for r in top_recs])
        avg_H_bottom   = _mean([r["seq_entropy"]   for r in bottom_recs])
        avg_len_top    = _mean([r["rollout_length"] for r in top_recs])
        avg_len_bottom = _mean([r["rollout_length"] for r in bottom_recs])

        top_label    = f"Top {int(args.top_ratio*100)}%"
        bottom_label = f"Bottom {int((1-args.top_ratio)*100)}%"

        print(f"\n  [{score_name}]")
        print(f"  {'Group':<14}  {'n':>5}  {'acc':>7}  {'avg_kl':>8}  {'avg_ent':>8}  {'avg_len':>8}")
        print(f"  {'-' * 58}")
        print(f"  {top_label:<14}  {len(top_recs):>5}  "
              f"{_fmt(acc_top, '2f'):>7}  "
              f"{_fmt(avg_kl_top):>8}  {_fmt(avg_H_top):>8}  {_fmt(avg_len_top, '0f'):>8}")
        print(f"  {bottom_label:<14}  {len(bottom_recs):>5}  "
              f"{_fmt(acc_bottom, '2f'):>7}  "
              f"{_fmt(avg_kl_bottom):>8}  {_fmt(avg_H_bottom):>8}  {_fmt(avg_len_bottom, '0f'):>8}")
        print(f"  {'All':<14}  {len(valid):>5}  "
              f"{_fmt(acc_all, '2f'):>7}  "
              f"{_fmt(_mean(K_raw)):>8}  {_fmt(_mean(H_raw)):>8}  {_fmt(_mean(L_raw), '0f'):>8}")

        if acc_top is not None and acc_bottom is not None:
            delta = acc_top - acc_bottom
            sign  = "+" if delta >= 0 else ""
            print(f"  Δacc (top - bottom) = {sign}{delta:.4f}  ({sign}{delta*100:.2f} pp)")

        summary["scores"][score_name] = {
            "top": {
                "n": len(top_recs), "evaluable": ev_top, "correct": cor_top,
                "accuracy": acc_top,
                "avg_kl_loss": avg_kl_top, "avg_entropy": avg_H_top, "avg_length": avg_len_top,
            },
            "bottom": {
                "n": len(bottom_recs), "evaluable": ev_bottom, "correct": cor_bottom,
                "accuracy": acc_bottom,
                "avg_kl_loss": avg_kl_bottom, "avg_entropy": avg_H_bottom, "avg_length": avg_len_bottom,
            },
            "delta_accuracy": (
                acc_top - acc_bottom if (acc_top is not None and acc_bottom is not None) else None
            ),
        }

    # ----------------------------------------------------------------
    # Quadrant breakdown (H vs K median split)
    # ----------------------------------------------------------------
    print(f"\n{'=' * 70}")
    print("R3-AFFINITY  —  Quadrant Distribution (H_norm × K_norm median split)")
    print(f"{'=' * 70}")

    n_v = len(valid)
    sorted_H = sorted(H_norm)
    sorted_K = sorted(K_norm)
    med_H = (sorted_H[(n_v - 1) // 2] + sorted_H[n_v // 2]) / 2
    med_K = (sorted_K[(n_v - 1) // 2] + sorted_K[n_v // 2]) / 2
    print(f"  median H_norm={med_H:.4f},  median K_norm={med_K:.4f}\n")

    quad_defs = [
        ("R3", "low_H",  "high_K", "低H + 高K  (确信被纠正, 最有价值)  ← R3"),
        ("R1", "high_H", "high_K", "高H + 高K  (探索+被纠正)            R1"),
        ("R4", "low_H",  "low_K",  "低H + 低K  (已学会, 无价值)         R4"),
        ("R2", "high_H", "low_K",  "高H + 低K  (双方都不确定)           R2"),
    ]

    quad_buckets = {k: [] for k, *_ in quad_defs}
    for i, r in enumerate(valid):
        lh = "low_H"  if H_norm[i] <= med_H else "high_H"
        lk = "high_K" if K_norm[i] >  med_K else "low_K"
        for rname, h_key, k_key, _ in quad_defs:
            if lh == h_key and lk == k_key:
                quad_buckets[rname].append(r)
                break

    quad_summary = {}
    for rname, _, _, label in quad_defs:
        recs  = quad_buckets[rname]
        n_q   = len(recs)
        ev    = [r for r in recs if r.get("correct") is not None]
        cor   = sum(1 for r in ev if r["correct"])
        acc   = cor / len(ev) if ev else None
        avg_s = _mean([r["S_R3"]          for r in recs])
        avg_k = _mean([r["kl_loss"]        for r in recs])
        avg_h = _mean([r["seq_entropy"]    for r in recs])
        avg_l = _mean([r["rollout_length"] for r in recs])

        print(f"  {label}")
        print(f"    n={n_q:4d}  acc={_fmt(acc, '4f')}  "
              f"avg_S_R3={_fmt(avg_s)}  avg_kl={_fmt(avg_k)}  "
              f"avg_H={_fmt(avg_h)}  avg_len={_fmt(avg_l, '0f')}")

        quad_summary[rname] = {
            "label": label, "n": n_q, "evaluable": len(ev), "correct": cor,
            "accuracy": acc, "avg_S_R3": avg_s,
            "avg_kl": avg_k, "avg_entropy": avg_h, "avg_length": avg_l,
        }

    summary["quadrant_r3"] = {
        "median_H_norm": med_H,
        "median_K_norm": med_K,
        "quads": quad_summary,
    }

    # ----------------------------------------------------------------
    # Write outputs
    # ----------------------------------------------------------------
    with open(out_jsonl, "w") as f:
        for r in valid:
            f.write(json.dumps(r) + "\n")

    with open(out_summary, "w") as f:
        json.dump(summary, f, indent=2)

    print(f"\n{'=' * 70}")
    print(f"Scored records -> {out_jsonl}")
    print(f"Summary        -> {out_summary}")
    print(f"{'=' * 70}")


if __name__ == "__main__":
    main()
