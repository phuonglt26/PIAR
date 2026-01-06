# run_p5_attr_only.py
# -*- coding: utf-8 -*-
import argparse, json, os, re, itertools
from typing import List, Dict, Any, Optional, Tuple
import torch
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
from torch.utils.data import DataLoader

# ---- Your dataset + collator (kept) ----
from prompt_loader import (
    MultiPromptSpanCollator,
    JsonlSeq2SeqMultiPromptSpanDataset,
)
from helper_func import print_progress


def pearson_corr(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    a = (a - a.mean()) / (a.std() + 1e-12)
    b = (b - b.mean()) / (b.std() + 1e-12)
    return (a * b).mean()


def spearman_corr(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    ar = a.argsort().argsort().float()
    br = b.argsort().argsort().float()
    return pearson_corr(ar, br)


def kendall_tau_corr(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    # O(n^2) but fine for attribution vectors
    n = a.numel()
    ai = a.repeat(n).view(n, n)
    aj = ai.t()
    bi = b.repeat(n).view(n, n)
    bj = bi.t()
    mask = torch.triu(torch.ones(n, n, device=a.device, dtype=torch.bool), diagonal=1)
    s = torch.sign((ai - aj) * (bi - bj))[mask]
    return s.mean()  # in [-1, 1]


def _prompt_ids_from_batch_keys(batch: Dict[str, Any]) -> List[int]:
    ids, pat = [], re.compile(r"^input_ids_(\d+)$")
    for k in batch.keys():
        m = pat.match(k)
        if m:
            ids.append(int(m.group(1)))
    ids = sorted(set(ids))
    assert ids, "No 'input_ids_{i}' keys in batch."
    return ids


@torch.no_grad()
def _decode_best_or_match(
    model, tokenizer, item_candidate, input_ids, attention_mask, num_beams, max_length
):
    gen_out = model.generate(
        input_ids=input_ids,
        attention_mask=attention_mask,
        max_length=max_length,
        num_beams=num_beams,
        num_return_sequences=num_beams,
        output_hidden_states=False,
        output_attentions=False,
        return_dict_in_generate=True,
    )
    best_ids = gen_out.sequences[0]
    if item_candidate:
        norm_cands = {c.strip().lower(): c for c in item_candidate}
        for g in gen_out.sequences:
            txt = (
                tokenizer.decode(
                    g, skip_special_tokens=True, clean_up_tokenization_spaces=False
                )
                .strip()
                .lower()
            )
            if txt in norm_cands:
                best_ids = g
                break
    return best_ids


def get_attcat_for_prompt(
    model,
    tokenizer,
    item_candidate: List[str],
    input_ids: torch.Tensor,  # [1, S]
    attention_mask: torch.Tensor,  # [1, S]
    num_beams: int = 1,
    max_length: int = 128,
    use_gold: bool = False,
    labels_gold: Optional[torch.Tensor] = None,  # [1, L] if use_gold
) -> torch.Tensor:
    """
    Return ATT-CAT over encoder tokens: grad*hidden per token, weighted by mean encoder attention.
    Output: [1, S] tensor.
    """
    assert (
        input_ids.shape[0] == 1 and attention_mask.shape[0] == 1
    ), "Batch=1 only for attribution."

    device = next(model.parameters()).device

    # Build decoder inputs
    if use_gold:
        assert labels_gold is not None and labels_gold.shape[0] == 1
        if hasattr(model, "prepare_decoder_input_ids_from_labels"):
            decoder_input_ids = model.prepare_decoder_input_ids_from_labels(
                labels=labels_gold.to(device)
            )
        else:
            dsid = model.config.decoder_start_token_id
            labels_ = labels_gold.to(device)
            shift = torch.full(
                (labels_.size(0), 1), dsid, dtype=labels_.dtype, device=labels_.device
            )
            decoder_input_ids = torch.cat([shift, labels_[:, :-1]], dim=1)
        gold_targets = labels_gold[:, 1:].contiguous().to(device)
    else:
        best_ids = _decode_best_or_match(
            model,
            tokenizer,
            item_candidate,
            input_ids,
            attention_mask,
            num_beams,
            max_length,
        )
        if hasattr(model, "prepare_decoder_input_ids_from_labels"):
            decoder_input_ids = model.prepare_decoder_input_ids_from_labels(
                labels=best_ids.unsqueeze(0).to(device)
            )
        else:
            dsid = model.config.decoder_start_token_id
            labels_ = best_ids.unsqueeze(0).to(device)
            shift = torch.full(
                (labels_.size(0), 1), dsid, dtype=labels_.dtype, device=labels_.device
            )
            decoder_input_ids = torch.cat([shift, labels_[:, :-1]], dim=1)
        gold_targets = None

    # Forward with attentions & hidden states
    outputs = model(
        input_ids=input_ids.to(device),
        attention_mask=attention_mask.to(device),
        decoder_input_ids=decoder_input_ids.to(device),
        output_hidden_states=True,
        output_attentions=True,
        return_dict=True,
    )

    model.zero_grad(set_to_none=True)

    # Encoder tensors
    enc_hs = outputs.encoder_hidden_states[1:]  # drop embeddings
    enc_atts = outputs.encoder_attentions  # list[L](B,H,S,S)
    num_layers = len(enc_atts)
    for l in range(num_layers):
        enc_hs[l].retain_grad()

    # Scalar objective for gradient
    logits = outputs.logits[0]  # [L_dec, V]
    if use_gold:
        pad_id = model.config.pad_token_id or 0
        tgt = gold_targets.clone()
        tgt[tgt < 0] = 0
        valid = (gold_targets != pad_id).squeeze(0)
        gold_logits = logits.gather(-1, tgt.squeeze(0).unsqueeze(-1)).squeeze(-1)
        scalar = gold_logits[valid].sum()
    else:
        top_pos = torch.argmax(logits, dim=-1)
        scalar = logits.gather(-1, top_pos.unsqueeze(-1)).sum()

    scalar.backward(retain_graph=False)

    # Grad*Act × mean attention over heads & queries
    cat_layers = {}
    for blk_id in range(num_layers):
        hs_grad = enc_hs[blk_id].grad
        assert hs_grad is not None, f"hs_grad must be NOT None for block {blk_id}."
        att = enc_atts[blk_id].mean(dim=1).mean(dim=2)  # Averaged attention weights
        # Gradient * Activation method
        cat_layer = (hs_grad * enc_hs[blk_id]).sum(dim=-1)
        cat_layer = cat_layer * att
        cat_layers[blk_id] = cat_layer

    # Optional normalization
    cat = sum(cat_layers.values())
    return cat  # [1, S]


def aggregate_items_from_mask(
    attr_tok: torch.Tensor, item_mask: torch.Tensor
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Variant that keeps sign and uses mean per item without final normalization.
    """
    if item_mask is None:
        raise ValueError("item_mask is missing.")

    masked = attr_tok.unsqueeze(1) * item_mask.float()  # [B, I, S]
    token_sum = masked.sum(dim=2)  # [B, I]
    valid = item_mask.any(dim=2)
    return token_sum, valid


def compute_attr_and_consistency(
    model,
    tokenizer,
    dataset,
    out_path: str,
    ks=(5, 10),
    topk: int = 10,
    num_beams: int = 1,
    max_length: int = 128,
    use_gold: bool = False,
    rank_metric: str = "spearman",
    batch_size: int = 1,
    num_examples: int = None,
    test_prompt_id_list: Optional[List[int]] = [1, 2, 3, 4, 5, 8, 9],
):
    device = next(model.parameters()).device
    model.eval()
    dl = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        collate_fn=MultiPromptSpanCollator(tokenizer, test_prompt_id_list),
    )

    ks = sorted(set(int(k) for k in ks if int(k) > 0))
    total = 0
    jacc_sum = {K: 0.0 for K in ks}
    l2_sum = 0.0
    js_sum = 0.0
    rank_sum = 0.0
    num_rank = 0
    num_pairs_total = 0
    per_prompt_exist = {}

    os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as f:
        base = 0
        for batch in dl:
            prompt_ids = _prompt_ids_from_batch_keys(batch)
            B = batch[f"input_ids_{prompt_ids[0]}"].size(0)
            if num_examples:
                if total >= num_examples:
                    break
            for b in range(B):
                src = dataset.samples[base + b]
                # candidates = src.get("candidates", [])
                candidates = None

                per_prompt_attr_vec = {}
                per_prompt_top = {}
                vec_len = None
                token_level = False

                for pid in prompt_ids:
                    ids = batch[f"input_ids_{pid}"][b : b + 1].to(device)  # [1,S]
                    msk = batch[f"attention_mask_{pid}"][b : b + 1].to(device)  # [1,S]
                    labels_gold = None
                    if (
                        use_gold
                        and ("labels" in batch)
                        and (batch["labels"] is not None)
                    ):
                        labels_gold = batch["labels"][b : b + 1].to(device)

                    attr_tok = get_attcat_for_prompt(
                        model,
                        tokenizer,
                        candidates,
                        ids,
                        msk,
                        num_beams=num_beams,
                        max_length=max_length,
                        use_gold=use_gold,
                        labels_gold=labels_gold,
                    )  # [1,S]

                    item_mask = batch.get(f"item_mask_{pid}", None)
                    if item_mask is not None:
                        im = item_mask[b : b + 1].to(device)  # [1,I,S]
                        attr_item, _ = aggregate_items_from_mask(attr_tok, im)
                        vec = attr_item.squeeze(0)  # [I]
                        token_level = False
                    else:
                        vec = attr_tok.squeeze(0)  # [S]
                        token_level = True

                    s = float(vec.sum().item())
                    vec = vec / (s if s > 1e-12 else 1.0)

                    per_prompt_attr_vec[pid] = vec
                    vec_len = vec_len or vec.numel()
                    per_prompt_exist[pid] = per_prompt_exist.get(pid, 0) + 1

                k_out = min(int(topk), int(vec_len or 0))
                for pid, vec in per_prompt_attr_vec.items():
                    order = torch.argsort(vec, descending=True)
                    top_idx = order[:k_out].tolist()
                    per_prompt_top[pid] = [
                        (int(i), float(vec[i].item())) for i in top_idx
                    ]

                if len(prompt_ids) >= 2:
                    for pa, pb in itertools.combinations(prompt_ids, 2):
                        va, vb = per_prompt_attr_vec[pa], per_prompt_attr_vec[pb]
                        L = min(va.numel(), vb.numel())
                        va, vb = va[:L], vb[:L]

                        l2_sum += float(torch.norm(va - vb, p=2).item())
                        p_ = torch.clamp(va, min=1e-8)
                        q_ = torch.clamp(vb, min=1e-8)
                        m_ = 0.5 * (p_ + q_)
                        js_pair = 0.5 * (
                            torch.sum(p_ * (p_.log() - m_.log()))
                            + torch.sum(q_ * (q_.log() - m_.log()))
                        )
                        js_sum += float(js_pair.item())

                        if rank_metric == "pearson":
                            rc = pearson_corr(va, vb)
                        elif rank_metric == "kendall":
                            rc = kendall_tau_corr(va, vb)
                        else:
                            rc = spearman_corr(va, vb)
                        if rc is not None and torch.isfinite(rc):
                            rank_sum += float(rc.item())
                            num_rank += 1

                        for K in ks:
                            A = set([i for i, _ in per_prompt_top[pa][:K]])
                            Bset = set([i for i, _ in per_prompt_top[pb][:K]])
                            denomJ = max(1, len(A | Bset))
                            jacc_sum[K] += len(A & Bset) / denomJ

                        num_pairs_total += 1

                dump = {
                    "input_idx": src.get("input_1", None),
                    "level": "item" if not token_level else "token",
                    "top_attr_by_prompt": {
                        str(pid): [
                            {"idx": int(i), "score": float(s)}
                            for i, s in per_prompt_top[pid]
                        ]
                        for pid in prompt_ids
                    },
                }
                # f.write(json.dumps(dump, ensure_ascii=False) + "\n")
                total += 1
                if num_examples:
                    print_progress(current=total, total=num_examples)
                else:
                    print_progress(current=total, total=len(dl))
            base += B

    metrics = {
        "total": total,
        "ATTR_CONSISTENCY": {
            **{
                f"mean_Jaccard@{K}": (jacc_sum[K] / max(1, num_pairs_total)) for K in ks
            },
            "mean_L2_attr": (l2_sum / max(1, num_pairs_total)),
            "mean_JS_attr": (js_sum / max(1, num_pairs_total)),
            "mean_rank_corr": (rank_sum / max(1, num_rank)) if num_rank > 0 else 0.0,
            "rank_metric": rank_metric,
        },
        "coverage_by_prompt": {
            str(pid): per_prompt_exist.get(pid, 0) for pid in per_prompt_exist
        },
    }

    with open(
        out_path.replace(".jsonl", "_metrics.jsonl"), "a", encoding="utf-8"
    ) as mf:
        mf.write(json.dumps(metrics, ensure_ascii=False) + "\n")

    print(f"✅ Wrote {metrics['total']} multiprompt attributions to {out_path}")
    for K in ks:
        print(
            f"Mean Jaccard@{K}: {metrics['ATTR_CONSISTENCY'][f'mean_Jaccard@{K}']:.4f}"
        )
    print("Mean L2(attr)  :", f"{metrics['ATTR_CONSISTENCY']['mean_L2_attr']:.6f}")
    print("Mean JS(attr)  :", f"{metrics['ATTR_CONSISTENCY']['mean_JS_attr']:.6f}")
    print(
        f"Mean {metrics['ATTR_CONSISTENCY']['rank_metric']} :",
        f"{metrics['ATTR_CONSISTENCY']['mean_rank_corr']:.6f}",
    )


def parse_args():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model_name", type=str, default="makitanikaze/P5")
    ap.add_argument(
        "--test_file",
        type=str,
        required=True,
        help="JSONL multiprompt dataset (eval only)",
    )
    ap.add_argument("--output_dir", type=str, default="./p5_attr_eval")
    ap.add_argument("--max_input_length", type=int, default=256)
    ap.add_argument("--max_target_length", type=int, default=16)
    ap.add_argument("--max_candidate_length", type=int, default=8)
    ap.add_argument("--generation_max_length", type=int, default=32)
    ap.add_argument("--generation_num_beams", type=int, default=1)
    ap.add_argument("--attr_metrics_k", type=str, default="5,10")
    ap.add_argument("--attr_topk", type=int, default=10)
    ap.add_argument("--attr_use_gold", action="store_true")
    ap.add_argument(
        "--attr_rank_corr",
        type=str,
        default="spearman",
        choices=["spearman", "pearson", "kendall"],
    )
    ap.add_argument(
        "--batch_size", type=int, default=1, help="keep 1 for strict token attribution"
    )
    ap.add_argument(
        "--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu"
    )
    ap.add_argument("--num_examples", type=int, default=None)
    ap.add_argument("--test_prompt_id_list", type=json.loads, default=None)

    return ap.parse_args()


def _has_model_weights(dir_):
    if not os.path.isdir(dir_):
        return False
    for fn in ["pytorch_model.bin", "model.safetensors", "pytorch_model.pt"]:
        if os.path.isfile(os.path.join(dir_, fn)):
            return True
    return False


def main():
    args = parse_args()
    os.makedirs(args.output_dir, exist_ok=True)
    device = torch.device("cuda")
    if torch.cuda.is_available():
        device = torch.device("cuda")
    else:
        raise EnvironmentError(
            "CUDA is not available. A GPU is required to run this script."
        )

    tokenizer = AutoTokenizer.from_pretrained(args.model_name, use_fast=True)
    if tokenizer.pad_token is None and tokenizer.eos_token is not None:
        tokenizer.pad_token = tokenizer.eos_token

    load_trained = _has_model_weights(args.output_dir)
    print(load_trained)
    if load_trained:
        print(f"🔄 Found trained weights in {args.output_dir} — loading them.")
        model = AutoModelForSeq2SeqLM.from_pretrained(args.output_dir)
    else:
        print(f"🆕 Loading base model: {args.model_name}")
        model = AutoModelForSeq2SeqLM.from_pretrained(args.model_name)

    test_ds = JsonlSeq2SeqMultiPromptSpanDataset(
        args.test_file,
        tokenizer,
        max_input_length=args.max_input_length,
        max_target_length=args.max_target_length,
        max_candidate_length=args.max_candidate_length,
        prompt_id_list=args.test_prompt_id_list,
    )

    ks_attr = [
        int(k) for k in str(args.attr_metrics_k).split(",") if k.strip().isdigit()
    ]
    out_attr = os.path.join(args.output_dir, "test_attr_multiprompt.jsonl")
    model.to(device)
    compute_attr_and_consistency(
        model=model,
        tokenizer=tokenizer,
        dataset=test_ds,
        out_path=out_attr,
        ks=ks_attr,
        topk=int(args.attr_topk),
        num_beams=int(args.generation_num_beams),
        max_length=int(args.generation_max_length or 128),
        use_gold=bool(args.attr_use_gold),
        rank_metric=args.attr_rank_corr,
        batch_size=max(1, int(args.batch_size)),
        num_examples=args.num_examples,
        test_prompt_id_list=args.test_prompt_id_list,
    )


if __name__ == "__main__":
    main()
