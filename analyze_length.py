"""
Length & entropy & KL-loss analysis on base model rollouts (vllm inference + hf forward).

For each of the first N problems in the training set:
  1. Generate one rollout via vllm (max_tokens=4096)
  2. Extract \\boxed{} answer and check correctness
  3. Record vllm-side stats: rollout_length, seq_entropy (top-k approx), truncated
  4. [KL phase] Run HF forward passes:
       student: base_model( student_prompt + rollout )
       teacher: base_model( teacher_prompt[with y*] + rollout )
     Compute forward KL(p_teacher || p_student) clipped at jsd_token_clip.
  5. Record per-sample: kl_loss, clip_ratio (from HF logits)

Output:
  <output_dir>/length_analysis.jsonl        — one JSON line per sample
  <output_dir>/length_analysis_summary.json — aggregate stats + correlations + quadrant analysis

Usage:
    python analyze_length.py \\
        --model_name_or_path /data0/shared/Qwen3-1.7B \\
        --output_dir /data0/siyanz/opsd/length_analysis \\
        --max_tokens 4096 \\
        --logprobs_top_k 20 \\
        --num_samples 3200 \\
        --tensor_parallel_size 4 \\
        --gpu_memory_utilization 0.85 \\
        --temperature 1.1 \\
        --top_p 0.95 \\
        --top_k 20 \\
        --jsd_token_clip 0.05 \\
        --kl_batch_size 4

    # To skip KL computation (faster, no --compute_kl flag):
    python analyze_length.py --model_name_or_path ... --skip_kl
"""

import argparse
import gc
import json
import math
import os
import re

import torch
import torch.nn.functional as F
from datasets import load_dataset
from transformers import AutoModelForCausalLM, AutoTokenizer
from vllm import LLM, SamplingParams


# ---------------------------------------------------------------------------
# Answer helpers
# ---------------------------------------------------------------------------

def _extract_boxed_answer(text: str):
    idx = text.rfind(r"\boxed{")
    if idx == -1:
        return None
    start = idx + len(r"\boxed{")
    depth = 1
    i = start
    while i < len(text) and depth > 0:
        if text[i] == "{":
            depth += 1
        elif text[i] == "}":
            depth -= 1
        i += 1
    return text[start : i - 1] if depth == 0 else None


def _normalize_math_answer(ans: str) -> str:
    if not ans:
        return ""
    ans = ans.strip()
    m = re.search(r"^\\text\{(.+?)\}$", ans)
    if m:
        ans = m.group(1).strip()
    ans = ans.replace("tfrac", "frac").replace("dfrac", "frac")
    ans = ans.replace("\\left", "").replace("\\right", "")
    ans = ans.replace(" ", "")
    try:
        f = float(ans)
        if abs(f - round(f)) < 1e-9:
            return str(int(round(f)))
        return f"{f:g}"
    except ValueError:
        pass
    return ans.lower()


# ---------------------------------------------------------------------------
# Entropy from vllm logprobs (top-k approximation)
# ---------------------------------------------------------------------------

def _seq_entropy_from_logprobs(logprobs_list) -> float:
    """
    logprobs_list: list of dicts per token, each dict maps token_id -> Logprob.
    For each token, compute H = -sum_k p_k * log(p_k) over top-k entries
    after renormalizing to a valid distribution.
    Returns mean entropy over the sequence.
    """
    if not logprobs_list:
        return 0.0

    token_entropies = []
    for token_logprobs in logprobs_list:
        if not token_logprobs:
            continue
        lps = [lp.logprob for lp in token_logprobs.values()]
        max_lp = max(lps)
        exps = [math.exp(lp - max_lp) for lp in lps]
        z = sum(exps)
        probs = [e / z for e in exps]
        h = -sum(p * math.log(p + 1e-12) for p in probs)
        token_entropies.append(h)

    return sum(token_entropies) / len(token_entropies) if token_entropies else 0.0


# ---------------------------------------------------------------------------
# Prompt builders
# ---------------------------------------------------------------------------

TRANSITION_PROMPT = (
    "\n\nAfter reading the reference solution above, make sure you truly understand "
    "the key reasoning steps. Now, using your own words and independent reasoning, "
    "derive the same final answer to the problem above. "
)


def build_student_prompt(tokenizer, problem: str) -> str:
    msg = f"Problem: {problem}\n\nPlease reason step by step, and put your final answer within \\boxed{{}}."
    messages = [{"role": "user", "content": msg}]
    return tokenizer.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True, enable_thinking=False
    )


def build_teacher_prompt(tokenizer, problem: str, solution: str) -> str:
    msg = (
        f"Problem: {problem}\n\n"
        f"Here is a reference solution to this problem:\n"
        f"=== Reference Solution Begin ===\n{solution}\n=== Reference Solution End ===\n"
        f"{TRANSITION_PROMPT}\n"
        f"Please reason step by step, and put your final answer within \\boxed{{}}."
    )
    messages = [{"role": "user", "content": msg}]
    return tokenizer.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True, enable_thinking=True
    )


# ---------------------------------------------------------------------------
# KL loss computation (mirrors analyze_base.py / opsd_trainer.py)
# ---------------------------------------------------------------------------

def compute_kl_stats(
    student_logits: torch.Tensor,   # (T, V)  response tokens only
    teacher_logits: torch.Tensor,   # (T, V)  response tokens only
    temperature: float,
    token_clip: float | None,
) -> dict:
    """
    Forward KL: KL(p_teacher || p_student)

    Returns rollout-level scalars AND token-level arrays needed for
    D_eff and C_KL computation:
      kl_loss      : mean per-token KL (clipped)
      clip_ratio   : fraction of tokens clipped
      kl_per_token : list[float], raw per-token KL before clip  (T,)
      h_per_token  : list[float], per-token student entropy     (T,)
    """
    s_lp = F.log_softmax(student_logits / temperature, dim=-1)  # (T, V)
    t_lp = F.log_softmax(teacher_logits / temperature, dim=-1)  # (T, V)

    # per-token student entropy: H = -sum_v p_s * log p_s
    h_per_token = -(s_lp.exp() * s_lp).sum(dim=-1)  # (T,)

    # forward KL(p_T || p_S)
    kl_per_token_raw = F.kl_div(s_lp, t_lp, reduction="none", log_target=True).sum(dim=-1)  # (T,)

    if token_clip is not None and token_clip > 0:
        clip_ratio = (kl_per_token_raw > token_clip).float().mean().item()
        kl_per_token_clipped = kl_per_token_raw.clamp(max=token_clip)
    else:
        clip_ratio = 0.0
        kl_per_token_clipped = kl_per_token_raw

    kl_loss = kl_per_token_clipped.mean().item()

    return {
        "kl_loss":       kl_loss,
        "clip_ratio":    clip_ratio,
        "kl_per_token":  kl_per_token_raw.cpu().tolist(),   # unclipped, for C_KL
        "h_per_token":   h_per_token.cpu().tolist(),
    }


@torch.no_grad()
def compute_kl_for_sample(
    model,
    hf_tokenizer,
    device,
    problem: str,
    solution: str,
    ground_truth: str,
    resp_token_ids: list[int],
    temperature: float,
    token_clip: float | None,
) -> dict:
    """
    Single-sample forward pass computing both full-y* KL and answer-only KL,
    plus token-level statistics needed for D_eff and C_KL.

    Rollout-level scalars returned:
      kl_loss        : mean per-token KL, full-y*  (original OPSD, clipped)
      clip_ratio     : clip fraction for full-y*
      kl_answer_only : mean per-token KL, answer-only y*
      kl_noise       : kl_loss - kl_answer_only  (path-noise KL)
      snr            : kl_answer_only / kl_loss   (signal-to-noise ratio)

    Token-level arrays (for offline D_eff / C_KL computation):
      kl_per_token   : list[float], per-token full-y* KL        (T,)
      h_per_token    : list[float], per-token student entropy    (T,)
      d_eff          : scalar, mean(kl_n * h_n)  — effective gradient density
      c_kl           : scalar, mean(kl_n) / std(kl_n)  — KL coherence
      s_rollout      : scalar, d_eff * c_kl  — composite rollout quality score
    """
    resp_tensor = torch.tensor(resp_token_ids, dtype=torch.long, device=device)

    # --- student forward (shared for both teacher variants) ---
    s_prompt = build_student_prompt(hf_tokenizer, problem)
    s_enc = hf_tokenizer(s_prompt, return_tensors="pt", truncation=True).to(device)
    s_prompt_len = s_enc["input_ids"].shape[1]
    s_full_ids = torch.cat([s_enc["input_ids"][0], resp_tensor], dim=0).unsqueeze(0)
    s_attn = torch.ones_like(s_full_ids)
    s_out = model(input_ids=s_full_ids, attention_mask=s_attn)
    s_logits = s_out.logits[0, s_prompt_len - 1 : -1, :]  # (T_resp, V)

    def _teacher_kl(teacher_solution: str) -> dict | None:
        t_prompt = build_teacher_prompt(hf_tokenizer, problem, teacher_solution)
        t_enc = hf_tokenizer(t_prompt, return_tensors="pt", truncation=True).to(device)
        t_prompt_len = t_enc["input_ids"].shape[1]
        t_full_ids = torch.cat([t_enc["input_ids"][0], resp_tensor], dim=0).unsqueeze(0)
        t_attn = torch.ones_like(t_full_ids)
        t_out = model(input_ids=t_full_ids, attention_mask=t_attn)
        t_logits = t_out.logits[0, t_prompt_len - 1 : -1, :]  # (T_resp, V)
        min_len = min(s_logits.shape[0], t_logits.shape[0])
        if min_len == 0:
            return None
        return compute_kl_stats(
            s_logits[:min_len], t_logits[:min_len],
            temperature=temperature, token_clip=token_clip,
        )

    # --- teacher forward A: full solution as y* ---
    stats_full = _teacher_kl(solution)

    # --- teacher forward B: answer-only as y* ---
    stats_ans = _teacher_kl(ground_truth)

    null = {
        "kl_loss": None, "clip_ratio": None,
        "kl_answer_only": None, "kl_noise": None, "snr": None,
        "kl_per_token": None, "h_per_token": None,
        "d_eff": None, "c_kl": None, "s_rollout": None,
    }
    if stats_full is None or stats_ans is None:
        return null

    kl_full = stats_full["kl_loss"]
    kl_ans  = stats_ans["kl_loss"]
    kl_noise = kl_full - kl_ans
    snr = kl_ans / kl_full if kl_full > 1e-12 else None

    # --- token-level D_eff and C_KL (use full-y* KL and student entropy) ---
    import statistics as _stats
    kl_tok = stats_full["kl_per_token"]   # list[float], unclipped
    h_tok  = stats_full["h_per_token"]    # list[float]

    # D_eff = mean(kl_n * h_n)  — only positions where KL > 0 carry signal
    d_eff = sum(k * h for k, h in zip(kl_tok, h_tok)) / len(kl_tok) if kl_tok else None

    # C_KL = mean(kl_n) / std(kl_n)  — coherence: high = KL concentrated
    if len(kl_tok) >= 2:
        kl_mean = sum(kl_tok) / len(kl_tok)
        kl_std  = _stats.stdev(kl_tok)
        c_kl = kl_mean / kl_std if kl_std > 1e-12 else None
    else:
        c_kl = None

    s_rollout = d_eff * c_kl if (d_eff is not None and c_kl is not None) else None

    return {
        "kl_loss":        kl_full,
        "clip_ratio":     stats_full["clip_ratio"],
        "kl_answer_only": kl_ans,
        "kl_noise":       kl_noise,
        "snr":            snr,
        "kl_per_token":   kl_tok,
        "h_per_token":    h_tok,
        "d_eff":          d_eff,
        "c_kl":           c_kl,
        "s_rollout":      s_rollout,
    }


# ---------------------------------------------------------------------------
# Args
# ---------------------------------------------------------------------------

_DAPO_INSTRUCTION_PREFIX = (
    "Solve the following math problem step by step. "
    "The last line of your response should be of the form "
    "Answer: $Answer (without quotes) where $Answer is the answer to the problem.\n\n"
)
_DAPO_INSTRUCTION_SUFFIX = '\n\nRemember to put your answer on its own line after "Answer:".'


def _extract_problem_from_dapo(content: str) -> str:
    """Strip DAPO's fixed instruction wrapper, keep only the math problem text."""
    if content.startswith(_DAPO_INSTRUCTION_PREFIX):
        content = content[len(_DAPO_INSTRUCTION_PREFIX):]
    if content.endswith(_DAPO_INSTRUCTION_SUFFIX):
        content = content[:-len(_DAPO_INSTRUCTION_SUFFIX)]
    return content.strip()


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--model_name_or_path", required=True)
    p.add_argument("--output_dir", required=True)
    p.add_argument("--dataset_name", type=str, default="siyanzhao/Openthoughts_math_30k_opsd",
                   help="HuggingFace dataset name. Supports 'siyanzhao/Openthoughts_math_30k_opsd' "
                        "and 'BytedTsinghua-SIA/DAPO-Math-17k' style datasets.")
    p.add_argument("--dataset_split", type=str, default="train",
                   help="Dataset split to use (default: train).")
    # vllm generation
    p.add_argument("--max_tokens", type=int, default=4096)
    p.add_argument("--logprobs_top_k", type=int, default=20,
                   help="Number of top logprobs returned per token for entropy estimation.")
    p.add_argument("--num_samples", type=int, default=3200,
                   help="Number of training samples to process. -1 = all.")
    p.add_argument("--tensor_parallel_size", type=int, default=4)
    p.add_argument("--gpu_memory_utilization", type=float, default=0.85)
    p.add_argument("--temperature", type=float, default=1.1)
    p.add_argument("--top_p", type=float, default=0.95)
    p.add_argument("--top_k", type=int, default=20)
    p.add_argument("--seed", type=int, default=42)
    # KL computation
    p.add_argument("--skip_kl", action="store_true",
                   help="Skip KL computation (only length/entropy stats).")
    p.add_argument("--jsd_token_clip", type=float, default=0.05,
                   help="Per-token KL clip threshold. 0 = no clip.")
    p.add_argument("--kl_batch_size", type=int, default=1,
                   help="Samples per HF forward batch (1 = safest for memory).")
    p.add_argument("--torch_dtype", type=str, default="bfloat16")
    p.add_argument("--attn_implementation", type=str, default="flash_attention_2")
    return p.parse_args()


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    args = parse_args()
    os.makedirs(args.output_dir, exist_ok=True)
    output_path = os.path.join(args.output_dir, "length_analysis.jsonl")
    summary_path = os.path.join(args.output_dir, "length_analysis_summary.json")

    # ----------------------------------------------------------------
    # Load dataset
    # ----------------------------------------------------------------
    print(f"Loading dataset: {args.dataset_name}  (split={args.dataset_split})")
    dataset = load_dataset(args.dataset_name)[args.dataset_split]
    if args.num_samples > 0:
        dataset = dataset.select(range(min(args.num_samples, len(dataset))))
    print(f"Total samples: {len(dataset)}")

    # Detect dataset format by checking available columns
    sample = dataset[0]
    is_dapo = "prompt" in sample and isinstance(sample["prompt"], list)

    if is_dapo:
        # DAPO format: prompt is a list of chat messages; extract problem from content
        # reward_model is a dict: {"ground_truth": "34", "style": "..."}
        # solution field is a category label (e.g. "MATH"), not a reference solution
        problems = [
            _extract_problem_from_dapo(ex["prompt"][0]["content"])
            for ex in dataset
        ]
        ground_truths = [str(ex["reward_model"].get("ground_truth", "")) for ex in dataset]
        solutions = ground_truths  # y* = final answer string, same as EGSD training logic
    else:
        # OPSD format: problem / solution / Answer fields
        problems   = [ex["problem"]  for ex in dataset]
        solutions  = [ex["solution"] for ex in dataset]
        ground_truths = []
        for ex in dataset:
            gt = ex.get("Answer") or ex.get("answer") or ex.get("solution", "")
            ground_truths.append(str(gt))

    # ----------------------------------------------------------------
    # Phase 1: vLLM generation
    # ----------------------------------------------------------------
    print(f"\nPhase 1: vLLM generation  (max_tokens={args.max_tokens})")
    llm = LLM(
        model=args.model_name_or_path,
        tensor_parallel_size=args.tensor_parallel_size,
        gpu_memory_utilization=args.gpu_memory_utilization,
        seed=args.seed,
        trust_remote_code=True,
    )
    vllm_tokenizer = llm.get_tokenizer()

    sampling_params = SamplingParams(
        temperature=args.temperature,
        top_p=args.top_p,
        top_k=args.top_k,
        max_tokens=args.max_tokens,
        logprobs=args.logprobs_top_k,
    )

    vllm_prompts = [build_student_prompt(vllm_tokenizer, p) for p in problems]
    outputs = llm.generate(vllm_prompts, sampling_params)
    print(f"  [{len(outputs)}/{len(vllm_prompts)}] generation done")

    # Collect vllm-side records + keep token_ids for KL phase
    records = []
    all_resp_token_ids = []   # list of list[int], used in KL phase

    for i, output in enumerate(outputs):
        seq = output.outputs[0]
        rollout_text   = seq.text
        rollout_length = len(seq.token_ids)
        resp_token_ids = list(seq.token_ids)

        seq_entropy = _seq_entropy_from_logprobs(seq.logprobs)

        predicted = _extract_boxed_answer(rollout_text)
        boxed_extracted = predicted is not None
        gt = ground_truths[i]
        if boxed_extracted and gt:
            correct = _normalize_math_answer(predicted) == _normalize_math_answer(gt)
        else:
            correct = None

        record = {
            "sample_idx":      i,
            "boxed_extracted": boxed_extracted,
            "correct":         correct,
            "rollout_length":  rollout_length,
            "seq_entropy":     seq_entropy,
            "truncated":       rollout_length >= args.max_tokens,
            # rollout-level scalars (filled in Phase 2)
            "kl_loss":         None,   # full-y* KL (clipped mean)
            "kl_clip_ratio":   None,
            "kl_answer_only":  None,   # answer-only KL
            "kl_noise":        None,   # kl_loss - kl_answer_only
            "snr":             None,   # kl_answer_only / kl_loss
            # token-level derived scores (filled in Phase 2)
            "d_eff":           None,   # mean(kl_n * h_n): effective gradient density
            "c_kl":            None,   # mean(kl_n) / std(kl_n): KL coherence
            "s_rollout":       None,   # d_eff * c_kl: composite rollout quality
            # raw token arrays (filled in Phase 2, for offline analysis)
            "kl_per_token":    None,   # list[float], per-token full-y* KL
            "h_per_token":     None,   # list[float], per-token student entropy
        }
        records.append(record)
        all_resp_token_ids.append(resp_token_ids)

    # Free vllm GPU memory before loading HF model
    del llm
    gc.collect()
    torch.cuda.empty_cache()
    print("  vLLM released.")

    # ----------------------------------------------------------------
    # Phase 2: HF forward passes for KL loss
    # ----------------------------------------------------------------
    if not args.skip_kl:
        print(f"\nPhase 2: HF forward passes for KL loss  (batch_size={args.kl_batch_size})")
        dtype_map = {"bfloat16": torch.bfloat16, "float16": torch.float16, "float32": torch.float32}
        dtype  = dtype_map[args.torch_dtype]
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        hf_tokenizer = AutoTokenizer.from_pretrained(
            args.model_name_or_path, trust_remote_code=True
        )
        if hf_tokenizer.pad_token is None:
            hf_tokenizer.pad_token = hf_tokenizer.eos_token

        model = AutoModelForCausalLM.from_pretrained(
            args.model_name_or_path,
            torch_dtype=dtype,
            attn_implementation=args.attn_implementation,
            trust_remote_code=True,
        ).to(device)
        model.eval()

        token_clip = args.jsd_token_clip if args.jsd_token_clip > 0 else None

        for i in range(len(records)):
            kl_stats = compute_kl_for_sample(
                model=model,
                hf_tokenizer=hf_tokenizer,
                device=device,
                problem=problems[i],
                solution=solutions[i],
                ground_truth=ground_truths[i],
                resp_token_ids=all_resp_token_ids[i],
                temperature=args.temperature,
                token_clip=token_clip,
            )
            records[i]["kl_loss"]        = kl_stats["kl_loss"]
            records[i]["kl_clip_ratio"]  = kl_stats["clip_ratio"]
            records[i]["kl_answer_only"] = kl_stats["kl_answer_only"]
            records[i]["kl_noise"]       = kl_stats["kl_noise"]
            records[i]["snr"]            = kl_stats["snr"]
            records[i]["d_eff"]          = kl_stats["d_eff"]
            records[i]["c_kl"]           = kl_stats["c_kl"]
            records[i]["s_rollout"]      = kl_stats["s_rollout"]
            records[i]["kl_per_token"]   = kl_stats["kl_per_token"]
            records[i]["h_per_token"]    = kl_stats["h_per_token"]

            if (i + 1) % 50 == 0 or (i + 1) == len(records):
                print(f"  [{i + 1}/{len(records)}] KL done", flush=True)

        del model
        gc.collect()
        torch.cuda.empty_cache()
        print("  HF model released.")
    else:
        print("\nPhase 2: skipped (--skip_kl).")

    # ----------------------------------------------------------------
    # Write per-sample jsonl
    # ----------------------------------------------------------------
    with open(output_path, "w") as f:
        for r in records:
            f.write(json.dumps(r) + "\n")
    print(f"\nPer-sample results saved to {output_path}")

    # ----------------------------------------------------------------
    # Aggregate summary
    # ----------------------------------------------------------------
    total = len(records)
    boxed_extracted_count = sum(1 for r in records if r["boxed_extracted"])
    evaluable     = [r for r in records if r["correct"] is not None]
    correct_records = [r for r in evaluable if r["correct"]]
    wrong_records   = [r for r in evaluable if not r["correct"]]
    has_kl = any(r["kl_loss"] is not None for r in records)

    def _mean(vals):
        vals = [v for v in vals if v is not None]
        return sum(vals) / len(vals) if vals else None

    def _kl_mean(group):
        return _mean([r["kl_loss"] for r in group])

    def _snr_mean(group):
        return _mean([r["snr"] for r in group])

    def _noise_mean(group):
        return _mean([r["kl_noise"] for r in group])

    def _group_stats(group):
        return {
            "avg_rollout_length":  _mean([r["rollout_length"]  for r in group]),
            "avg_seq_entropy":     _mean([r["seq_entropy"]     for r in group]),
            "avg_kl_loss":         _kl_mean(group),
            "avg_kl_answer_only":  _mean([r["kl_answer_only"] for r in group]),
            "avg_kl_noise":        _noise_mean(group),
            "avg_snr":             _snr_mean(group),
            "avg_d_eff":           _mean([r["d_eff"]      for r in group]),
            "avg_c_kl":            _mean([r["c_kl"]       for r in group]),
            "avg_s_rollout":       _mean([r["s_rollout"]  for r in group]),
        }

    summary = {
        "total_samples": total,
        "boxed_extracted": boxed_extracted_count,
        "boxed_extraction_rate": boxed_extracted_count / total if total else 0,
        "evaluable_samples": len(evaluable),
        "correct_count": len(correct_records),
        "wrong_count": len(wrong_records),
        "accuracy": len(correct_records) / len(evaluable) if evaluable else None,
        "correct":  _group_stats(correct_records),
        "wrong":    _group_stats(wrong_records),
        "no_boxed": {"count": total - boxed_extracted_count,
                     **_group_stats([r for r in records if not r["boxed_extracted"]])},
    }

    # ----------------------------------------------------------------
    # Correlation: rollout_length vs seq_entropy  (non-truncated only)
    # ----------------------------------------------------------------
    def _pearson(xs, ys):
        n = len(xs)
        if n < 2:
            return None
        mx, my = sum(xs) / n, sum(ys) / n
        num = sum((x - mx) * (y - my) for x, y in zip(xs, ys))
        dx = math.sqrt(sum((x - mx) ** 2 for x in xs))
        dy = math.sqrt(sum((y - my) ** 2 for y in ys))
        return num / (dx * dy) if dx * dy > 0 else None

    def _spearman(xs, ys):
        n = len(xs)
        if n < 2:
            return None
        rank_x = {v: r + 1 for r, v in enumerate(sorted(range(n), key=lambda i: xs[i]))}
        rank_y = {v: r + 1 for r, v in enumerate(sorted(range(n), key=lambda i: ys[i]))}
        rx = [rank_x[i] for i in range(n)]
        ry = [rank_y[i] for i in range(n)]
        return _pearson(rx, ry)

    not_truncated         = [r for r in records         if not r["truncated"]]
    not_truncated_correct = [r for r in correct_records if not r["truncated"]]
    not_truncated_wrong   = [r for r in wrong_records   if not r["truncated"]]
    not_truncated_no_box  = [r for r in records         if not r["boxed_extracted"] and not r["truncated"]]
    truncated_count       = sum(1 for r in records if r["truncated"])

    def _group_corr(group):
        if len(group) < 2:
            return None, None
        ls = [r["rollout_length"] for r in group]
        es = [r["seq_entropy"]    for r in group]
        return _pearson(ls, es), _spearman(ls, es)

    corr_all_p,     corr_all_s     = _group_corr(not_truncated)
    corr_correct_p, corr_correct_s = _group_corr(not_truncated_correct)
    corr_wrong_p,   corr_wrong_s   = _group_corr(not_truncated_wrong)
    corr_nobox_p,   corr_nobox_s   = _group_corr(not_truncated_no_box)

    correlation = {
        "note": "computed on non-truncated samples only (rollout_length < max_tokens)",
        "truncated_count":     truncated_count,
        "non_truncated_count": len(not_truncated),
        "all":      {"pearson": corr_all_p,     "spearman": corr_all_s,     "n": len(not_truncated)},
        "correct":  {"pearson": corr_correct_p, "spearman": corr_correct_s, "n": len(not_truncated_correct)},
        "wrong":    {"pearson": corr_wrong_p,   "spearman": corr_wrong_s,   "n": len(not_truncated_wrong)},
        "no_boxed": {"pearson": corr_nobox_p,   "spearman": corr_nobox_s,   "n": len(not_truncated_no_box)},
    }
    summary["correlation_length_vs_entropy"] = correlation

    # ----------------------------------------------------------------
    # Quadrant analysis: (length, entropy) × 4, plus kl_loss per quadrant
    # Split by median of NON-TRUNCATED samples
    # ----------------------------------------------------------------
    def _quadrant_stats(group, med_len, med_ent):
        quads = {
            "short_low":  [],   # length<=med, entropy<=med
            "short_high": [],   # length<=med, entropy>med
            "long_low":   [],   # length>med,  entropy<=med
            "long_high":  [],   # length>med,  entropy>med
        }
        for r in group:
            short = r["rollout_length"] <= med_len
            low   = r["seq_entropy"]    <= med_ent
            key   = ("short" if short else "long") + "_" + ("low" if low else "high")
            quads[key].append(r)

        result = {}
        for k, rs in quads.items():
            n  = len(rs)
            ev = [r for r in rs if r["correct"] is not None]
            cor = sum(1 for r in ev if r["correct"])
            kl_vals = [r["kl_loss"] for r in rs if r["kl_loss"] is not None]
            result[k] = {
                "count":        n,
                "correct_count": cor,
                "evaluable":    len(ev),
                "correct_rate": cor / len(ev) if ev else None,
                "avg_length":   sum(r["rollout_length"] for r in rs) / n if n else None,
                "avg_entropy":  sum(r["seq_entropy"]    for r in rs) / n if n else None,
                "avg_kl_loss":  sum(kl_vals) / len(kl_vals) if kl_vals else None,
            }
        return result

    if len(not_truncated) >= 4:
        sorted_lens = sorted(r["rollout_length"] for r in not_truncated)
        sorted_ents = sorted(r["seq_entropy"]    for r in not_truncated)
        n_nt = len(not_truncated)
        med_len = (sorted_lens[(n_nt - 1) // 2] + sorted_lens[n_nt // 2]) / 2
        med_ent = (sorted_ents[(n_nt - 1) // 2] + sorted_ents[n_nt // 2]) / 2

        quad_all     = _quadrant_stats(not_truncated,         med_len, med_ent)
        quad_correct = _quadrant_stats(not_truncated_correct, med_len, med_ent)
        quad_wrong   = _quadrant_stats(not_truncated_wrong,   med_len, med_ent)

        quadrant_info = {
            "note": "non-truncated samples split by median length and median entropy",
            "median_length":  med_len,
            "median_entropy": med_ent,
            "all":     quad_all,
            "correct": quad_correct,
            "wrong":   quad_wrong,
        }
        summary["quadrant_analysis"] = quadrant_info
    else:
        quadrant_info = None
        summary["quadrant_analysis"] = None

    with open(summary_path, "w") as f:
        json.dump(summary, f, indent=2)

    # ----------------------------------------------------------------
    # Print summary
    # ----------------------------------------------------------------
    def _fmt(v):
        return f"{v:.4f}" if v is not None else "N/A"

    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)
    print(f"Total samples:          {total}")
    print(f"Boxed extracted:        {boxed_extracted_count} ({summary['boxed_extraction_rate']:.1%})")
    print(f"Evaluable:              {len(evaluable)}")
    print(f"Accuracy (on evaluable):{summary['accuracy']:.1%}" if summary['accuracy'] is not None else "Accuracy: N/A")
    print()

    def _print_group(name, s):
        print(f"{name}:")
        v = s["avg_rollout_length"]
        print(f"  avg rollout length:  {v:.1f}" if v is not None else "  avg rollout length: N/A")
        v = s["avg_seq_entropy"]
        print(f"  avg seq entropy:     {v:.4f}" if v is not None else "  avg seq entropy: N/A")
        if has_kl:
            v = s["avg_kl_loss"]
            print(f"  avg kl_full:         {v:.4f}" if v is not None else "  avg kl_full: N/A")
            v = s["avg_kl_answer_only"]
            print(f"  avg kl_answer_only:  {v:.4f}" if v is not None else "  avg kl_answer_only: N/A")
            v = s["avg_kl_noise"]
            print(f"  avg kl_noise:        {v:.4f}" if v is not None else "  avg kl_noise: N/A")
            v = s["avg_snr"]
            print(f"  avg snr:             {v:.4f}" if v is not None else "  avg snr: N/A")
            v = s["avg_d_eff"]
            print(f"  avg d_eff:           {v:.6f}" if v is not None else "  avg d_eff: N/A")
            v = s["avg_c_kl"]
            print(f"  avg c_kl:            {v:.4f}" if v is not None else "  avg c_kl: N/A")
            v = s["avg_s_rollout"]
            print(f"  avg s_rollout:       {v:.6f}" if v is not None else "  avg s_rollout: N/A")

    _print_group(f"Correct ({len(correct_records)} samples)", summary["correct"])
    print()
    _print_group(f"Wrong ({len(wrong_records)} samples)", summary["wrong"])
    print()
    _print_group(f"No boxed ({summary['no_boxed']['count']} samples)", summary["no_boxed"])
    print()

    print("=" * 60)
    print("CORRELATION: rollout_length vs seq_entropy  (non-truncated)")
    print("=" * 60)
    print(f"  truncated={truncated_count}, non-truncated={len(not_truncated)}, max_tokens={args.max_tokens}")
    print(f"  All non-trunc ({len(not_truncated):>5}):  Pearson={_fmt(corr_all_p)},  Spearman={_fmt(corr_all_s)}")
    print(f"  Correct       ({len(not_truncated_correct):>5}):  Pearson={_fmt(corr_correct_p)},  Spearman={_fmt(corr_correct_s)}")
    print(f"  Wrong         ({len(not_truncated_wrong):>5}):  Pearson={_fmt(corr_wrong_p)},  Spearman={_fmt(corr_wrong_s)}")
    print(f"  No boxed      ({len(not_truncated_no_box):>5}):  Pearson={_fmt(corr_nobox_p)},  Spearman={_fmt(corr_nobox_s)}")
    print("=" * 60)

    # ----------------------------------------------------------------
    # Print quadrant table
    # ----------------------------------------------------------------
    if quadrant_info is not None:
        print()
        print("=" * 60)
        print("QUADRANT ANALYSIS (non-truncated samples)")
        print(f"  median length  = {med_len:.1f}")
        print(f"  median entropy = {med_ent:.4f}")
        print("=" * 60)

        def _qfmt_cell(q_data, key):
            d = q_data[key]
            acc = f"{d['correct_rate']:.1%}" if d["correct_rate"] is not None else "  N/A  "
            kl  = f"kl={d['avg_kl_loss']:.4f}" if (has_kl and d["avg_kl_loss"] is not None) else ""
            base = f"n={d['count']:4d}  acc={acc}  len={d['avg_length']:6.0f}  ent={d['avg_entropy']:.3f}"
            return f"{base}  {kl}" if kl else base

        col_w   = 56 if has_kl else 46
        row_sep = "+" + "-" * 14 + "+" + "-" * col_w + "+" + "-" * col_w + "+"
        header  = f"| {'':12s} | {'长度短 (short)':^{col_w-2}s} | {'长度长 (long)':^{col_w-2}s} |"

        def _row(label, q_data, key_short, key_long):
            cell_s = _qfmt_cell(q_data, key_short)
            cell_l = _qfmt_cell(q_data, key_long)
            return f"| {label:12s} | {cell_s:^{col_w-2}s} | {cell_l:^{col_w-2}s} |"

        for group_name, q_data in [("ALL", quad_all), ("CORRECT", quad_correct), ("WRONG", quad_wrong)]:
            print(f"\n  [{group_name}]")
            print(row_sep)
            print(header)
            print(row_sep)
            print(_row("entropy 低", q_data, "short_low",  "long_low"))
            print(row_sep)
            print(_row("entropy 高", q_data, "short_high", "long_high"))
            print(row_sep)
    else:
        print("\n  (not enough non-truncated samples for quadrant analysis)")

    print(f"\nSummary saved to {summary_path}")


if __name__ == "__main__":
    main()
