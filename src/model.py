import random
import json, math, os, itertools, re
from typing import List, Dict, Any, Optional, Tuple
import torch
import torch.nn.functional as F
from transformers import (
    Seq2SeqTrainer,
    PreTrainedModel,
)
import copy
from src.helper_func import grad_influence_stats, print_progress


class P5FairAttrTrainer(Seq2SeqTrainer):
    """
    Multi-prompt training with:
      - CE loss averaged over K prompts
      - Prediction consistency (JSD) across prompts
      - Attribution consistency (ATT-CAT) across prompts (encoder-side)
    Expect batch keys:
      - for i in 1..K: input_ids_i, attention_mask_i[, item_mask_i]
      - labels, cand_input_ids

    Custom knobs attached to `self` in main():
      - lambda_cons_attr
      - attr_metric in {'l2','mse','jsd'}
      - attr_relu: bool
      - attr_center in {'mince','allpairs'}  # pairing for both pred+attr if you wish
    """

    @staticmethod
    def get_attcat_for_prompt(
        model,
        input_ids: torch.Tensor,  # [1, S]
        attention_mask: torch.Tensor,  # [1, S]
        attr_norm: Optional[str] = False,
        is_encoder: bool = True,  # NEW: chọn encoder/decoder để tính grad
        gold_labels: Optional[torch.Tensor] = None,  # [1, L] nếu use_gold=True
    ) -> torch.Tensor | Tuple[torch.Tensor, torch.Tensor]:

        outputs = model(
            input_ids=input_ids.to("cuda"),
            labels=gold_labels,
            attention_mask=attention_mask.to("cuda"),
            output_hidden_states=True,
            output_attentions=True,
            return_dict=True,
        )

        if is_encoder:
            num_blocks = len(outputs.encoder_attentions)
            hidden_states = outputs.encoder_hidden_states
            hidden_states = hidden_states[1:]
            attentions = outputs.encoder_attentions
        else:
            raise NotImplementedError("Decoder attribution not implemented yet.")
        hs_all = []
        for hs in hidden_states:
            hs.retain_grad()
            hs_all.append(hs)
        logits = outputs.logits
        # (1) Build mask: 1 for valid labels, 0 for pad positions
        valid_mask = (gold_labels != -100).float()  # [B, L]

        # (2) For gather, replace pad labels with 0 (any valid index)
        safe_labels = gold_labels.clone()
        safe_labels[gold_labels == -100] = 0

        # (3) Gather logits at gold label positions: [B, L]
        gold_logits = logits.gather(2, safe_labels.unsqueeze(-1)).squeeze(-1)

        # (4) Mask out padding positions
        gold_logits_masked = gold_logits * valid_mask  # [B, L]

        # (5) Sum only valid positions → [B]
        sum_gold_logits = gold_logits_masked.sum()
        sum_gold_logits.backward(retain_graph=True)
        cat_layers = {}
        for blk_id in range(num_blocks):
            hs_grad = hs_all[blk_id].grad
            assert hs_grad is not None, f"hs_grad must be NOT None for block {blk_id}."
            att = (
                attentions[blk_id].mean(dim=1).mean(dim=2)
            )  # Averaged attention weights
            # Gradient * Activation method
            cat_layer = (hs_grad * hidden_states[blk_id]).sum(dim=-1)
            cat_layer = cat_layer * att
            cat_layers[blk_id] = cat_layer

        # Aggregated over layers -> [B, S]
        cat = sum(cat_layers.values())  # elementwise + over dict values

        if attr_norm == "relu":
            cat_expln = torch.relu(cat)  # [B, S]
        elif attr_norm is None:
            cat_expln = cat
        elif attr_norm == "sum_norm":
            denom = cat.abs().sum(dim=1, keepdim=True) + 1e-8  # [B, 1]
            cat_expln = cat / denom  # [B, S]
        else:
            raise ValueError(f"attr_norm {attr_norm} not implemented.")

        return cat_expln

    @staticmethod
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
        neg_sum = (attr_tok * (attr_tok < 0)).sum(dim=1)
        item_neg_sum = (token_sum * (token_sum < 0)).sum(dim=1)
        neg_ratio = item_neg_sum / neg_sum
        pos_sum = (attr_tok * (attr_tok > 0)).sum(dim=1)
        item_pos_sum = (token_sum * (token_sum > 0)).sum(dim=1)
        pos_ratio = item_pos_sum / pos_sum
        return token_sum, neg_ratio, pos_ratio


    @staticmethod
    def _prompt_ids_from_batch_keys(batch: Dict[str, Any]) -> List[int]:
        """Detect prompt ids via keys like 'input_ids_{i}'."""
        ids = []
        pat = re.compile(r"^input_ids_(\d+)$")
        for k in batch.keys():
            m = pat.match(k)
            if m:
                ids.append(int(m.group(1)))
        ids = sorted(set(ids))
        assert ids, "No 'input_ids_{i}' keys in batch."
        return ids


    def compute_loss(self, model, inputs, return_outputs: bool = False, **kwargs):
        batch = dict(inputs)
        cand_ids = batch.pop("cand_input_ids", None)
        labels = batch.get("labels")
        # check current epoch
        current_epoch = self.state.epoch

        # discover prompts present in batch
        prompt_ids = self._prompt_ids_from_batch_keys(batch)
        K = len(prompt_ids)

        # 1) Base CE loss = mean over prompts
        ce_losses = []
        out_first = None
        for pid in prompt_ids:
            ids = batch[f"input_ids_{pid}"]
            msk = batch[f"attention_mask_{pid}"]
            out = model(
                input_ids=ids, attention_mask=msk, labels=labels, return_dict=True
            )
            ce_losses.append(out.loss)
            if out_first is None:
                out_first = out
        loss_ce = torch.stack(ce_losses).mean()
        logs: Dict[str, float] = {}
        total_loss = loss_ce

        logs["loss_ce"] = float(loss_ce.detach().item())

        # knobs
        lam_attr = float(getattr(self, "lambda_cons_attr", 0.0) or 0.0)
        pair_mode = str(getattr(self, "attr_center", "mince"))  # dùng cho pred + attr
        if current_epoch is not None and current_epoch <= self.num_epoch_warmup:
            lam_attr = 0.0

        if model.training:
            # 2) Attribution consistency (ATT-CAT, encoder-side) with min-CE center per-example
            if lam_attr > 0.0 and cand_ids is not None and K >= 2:
                # ---- Cache CE per example cho mọi prompt (dùng chung cho pred/attr) ----
                cidx = ce_losses.index(min(ce_losses))
                item_attr_all_prompts = (
                    []
                )  # list of attr at layers of all prompts [num_prompts, B,I]; B bach size, I num items
                I_list = []
                neg_ratios = []
                pos_ratios = []

                # Gọi get_attcat_for_prompt THEO BATCH (không cắt [b:b+1])
                for pid in prompt_ids:
                    ids = batch[f"input_ids_{pid}"]
                    msk = batch[f"attention_mask_{pid}"]

                    attr_tok = self.get_attcat_for_prompt(
                        model,
                        input_ids=ids,  # [B,S]
                        attention_mask=msk,  # [B,S]
                        attr_norm=self.attr_norm,
                        gold_labels=labels,  # [B,T]
                    )  # [B,S] (đã chuẩn hoá theo hàng)

                    item_mask = batch.get(f"item_mask_{pid}", None)  # [B,I,S] hoặc None
                    if item_mask is None:
                        raise ValueError(
                            "item_mask_{'pid'} not in batch, please provide item_mask_{'prompt_idx'}."
                        )
                    # attr_tok là dict {layer: [B,S]}
                    if self.sum_attr_per_item:
                        attr_item, neg_ratio, pos_ratio = (
                            self.aggregate_items_from_mask(attr_tok, item_mask)
                        )  # [B,I] (I is number of items at sample with largest I in a batch)
                        I = attr_item.size(1)
                    else:
                        raise NotImplementedError
                    # print("ai", ai)
                    item_attr_all_prompts.append(attr_item)
                    neg_ratios.append(neg_ratio)
                    pos_ratios.append(pos_ratio)
                    I_list.append(I)
                # print("I_list", I_list)
                I_common = min(I_list) if I_list else 0
                if I_common > 0:
                    # print("I_common", I_common)

                    item_attr_all_prompts = [
                        t[:, :I_common] for t in item_attr_all_prompts
                    ]
                    item_attr_all_prompts = torch.stack(
                        item_attr_all_prompts
                    )  # [num_prompts, B, I]
                    neg_ratios = torch.stack(neg_ratios)
                    pos_ratios = torch.stack(pos_ratios)
                    pair_losses = []
                    if pair_mode == "random":
                        # use random center for each (user, items)
                        # cidx = int(center_idx_per_b[b].item())
                        cidx = random.randint(0, K - 1)
                        center = item_attr_all_prompts[cidx]  # [I]
                        if self.attr_metric == "mse":
                            raise NotImplementedError(
                                "Random mode with MSE not implemented yet."
                            )
                        elif self.attr_metric == "jsd":
                            raise NotImplementedError(
                                "Random mode with JSD not implemented yet."
                            )
                        elif self.attr_metric == "l1":  # l1
                            pl = item_attr_all_prompts - center
                            pl = pl.abs()
                        else:  # l2
                            raise NotImplementedError(
                                "Random mode with L2 not implemented yet."
                            )
                        pair_losses = pl
                    if pair_mode == "fixed":
                        # use random center for each (user, items)
                        # cidx = int(center_idx_per_b[b].item())
                        self.center_prompt_id = 3
                        cidx = self.center_prompt_id - 1
                        center = item_attr_all_prompts[cidx]  # [I]
                        if self.attr_metric == "mse":
                            raise NotImplementedError(
                                "Random mode with MSE not implemented yet."
                            )
                        elif self.attr_metric == "jsd":
                            raise NotImplementedError(
                                "Random mode with JSD not implemented yet."
                            )
                        elif self.attr_metric == "l1":  # l1
                            pl = item_attr_all_prompts - center
                            pl = pl.abs()
                        else:  # l2
                            raise NotImplementedError(
                                "Random mode with L2 not implemented yet."
                            )
                        pair_losses.append(pl)

                    elif pair_mode == "mince":
                        # use mince center for each (user, items)
                        # cidx = int(center_idx_per_b[b].item())
                        center = item_attr_all_prompts[cidx]  # [I]
                        if self.attr_metric == "mse":
                            raise NotImplementedError(
                                "Random mode with MSE not implemented yet."
                            )
                        elif self.attr_metric == "jsd":
                            raise NotImplementedError(
                                "Random mode with JSD not implemented yet."
                            )
                        elif self.attr_metric == "l1":  # l1
                            pl = item_attr_all_prompts - center
                            pl = pl.abs()
                            negl = neg_ratios - neg_ratios[cidx]
                            negl = negl.abs()
                            posl = pos_ratios - pos_ratios[cidx]
                            posl = posl.abs()

                        else:  # l2
                            raise NotImplementedError(
                                "Random mode with L2 not implemented yet."
                            )
                        pair_losses.append(pl)
                    elif pair_mode == "mean":
                        # use mince center for each (user, items)
                        # cidx = int(center_idx_per_b[b].item())
                        center = item_attr_all_prompts.mean(dim=0)  # [I]
                        if self.attr_metric == "mse":
                            raise NotImplementedError(
                                "Random mode with MSE not implemented yet."
                            )
                        elif self.attr_metric == "jsd":
                            raise NotImplementedError(
                                "Random mode with JSD not implemented yet."
                            )
                        elif self.attr_metric == "l1":  # l1
                            pl = item_attr_all_prompts - center
                            pl = pl.abs()
                        else:  # l2
                            raise NotImplementedError(
                                "Random mode with L2 not implemented yet."
                            )
                        pair_losses.append(pl)
                    else:
                        # use allpairs center for each (user, items)
                        pair_losses = []
                        for pid in range(len(item_attr_all_prompts)):
                            if self.attr_metric == "mse":
                                raise NotImplementedError(
                                    "Random mode with MSE not implemented yet."
                                )
                            elif self.attr_metric == "jsd":
                                raise NotImplementedError(
                                    "Random mode with JSD not implemented yet."
                                )
                            elif self.attr_metric == "l1":  # l1
                                pl = item_attr_all_prompts - item_attr_all_prompts[pid]
                                pl = pl.abs()
                                pl = pl[pl != 0].mean()
                            else:  # l2
                                raise NotImplementedError(
                                    "Random mode with L2 not implemented yet."
                                )
                            pair_losses.append(pl)

                    if pair_losses:
                        pair_losses = torch.stack(pair_losses)
                        loss_attr = pair_losses[pair_losses != 0].mean()
                        total_loss = total_loss + lam_attr * loss_attr
                        logs["loss_cons_attr"] = float(loss_attr.detach().item())
                    else:
                        logs["loss_cons_attr"] = 0.0
                else:
                    logs["loss_cons_attr"] = 0.0
            else:
                logs["loss_cons_attr"] = 0.0

            # if self.lambda_item_prompt > 0.0:
            #     pass

            # ---- log everything
            logs["loss_total"] = float(total_loss.detach().item())
            if self.check_gradient:
                grp = ["encoder.", "decoder.", "lm_head", "embed_tokens"]
                st_ce = grad_influence_stats(loss_ce, model, eps=1e-10, group_by=grp)
                logs["ce_ratio_used"] = st_ce["ratio_used"]
                logs["ce_ratio_nonzero"] = st_ce["ratio_nonzero"]
                logs["ce_grad_mean"] = st_ce["grad_norm_mean"]

                if lam_attr > 0:
                    st_attr = grad_influence_stats(
                        lam_attr * loss_attr, model, eps=1e-10, group_by=grp
                    )
                    logs["attr_ratio_used"] = st_attr["ratio_used"]
                    logs["attr_ratio_nonzero"] = st_attr["ratio_nonzero"]
                    logs["attr_grad_mean"] = st_attr["grad_norm_mean"]
            if self.log_per_sample:
                self.log(logs)

        if return_outputs:
            out = out_first
            out.loss = total_loss
            return total_loss, out
        else:
            return total_loss

    @staticmethod
    def _masked_softmax(
        scores: torch.Tensor, valid: torch.Tensor, dim: int = 1
    ) -> torch.Tensor:
        masked = torch.where(valid, scores, torch.full_like(scores, -1e9))
        return torch.softmax(masked, dim=dim)

    @staticmethod
    def _dcg_at_pos(pos0: int) -> float:
        return 1.0 / math.log2(pos0 + 2.0)

    @torch.no_grad()
    def generate_prediction_beam(
        self,
        dataset,
        tokenizer,
        out_path: str,
        generation_max_length: int = 16,
        generation_num_beams: int = 20,
        num_examples=None,
        ks=(1, 5, 10),
        topk=10,
    ):
        """
        Evaluation for multiprompt sequential recommendation using beam search.

        For each prompt:
        - Use model.generate with num_beams and num_return_sequences = topk
        - Decode topk beams as item strings
        - Rank them by beam score
        - Compute Hit@K, NDCG@K using the gold target string

        Cross prompt consistency:
        - Top1 agreement
        - Jaccard@K over sets of predicted items
        - L2 and JS divergence between normalized beam score distributions
            over the union of generated items for each prompt pair
        """

        import json, os, itertools
        from typing import Dict, List
        import torch

        ks = sorted(set(int(k) for k in ks if int(k) > 0))
        if getattr(self.args, "process_index", 0) not in (0, None):
            return {"total": 0}  # only rank 0 writes / logs

        model = getattr(self.model, "module", self.model)
        gc = getattr(model, "generation_config", None)

        # Fallbacks
        if gc is not None and getattr(gc, "max_length", None) is not None:
            default_max_len = gc.max_length
        else:
            default_max_len = 16

        if gc is not None and getattr(gc, "num_beams", None) is not None:
            default_num_beams = gc.num_beams
        else:
            default_num_beams = 20

        max_len = generation_max_length or default_max_len
        num_beams = generation_num_beams or default_num_beams

        # Ensure we can actually return topk beams
        num_return = min(topk, num_beams)

        dl = self.get_eval_dataloader(dataset)
        model.eval()

        pad_id = int(
            getattr(tokenizer, "pad_token_id", 0)
            or getattr(model.config, "pad_token_id", 0)
            or 0
        )

        # Accumulators
        total = 0
        per_prompt_exact: Dict[int, int] = {}  # EM per prompt
        per_prompt_hits: Dict[int, Dict[int, int]] = {}  # Hit@K per prompt
        per_prompt_ndcg: Dict[int, Dict[int, float]] = {}  # NDCG@K per prompt

        top1_agree_sum = 0
        jacc_sum = {K: 0.0 for K in ks}
        l2_sum = 0.0
        js_sum = 0.0
        pair_count = 0  # number of prompt pairs actually compared

        os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)

        def _softmax_scores_over_union(texts_a, scores_a, texts_b, scores_b, device):
            """
            Build aligned probability distributions p, q over the union of
            generated items from two prompts, based on their beam scores.
            """
            # Map item -> best score among its beams
            map_a = {}
            map_b = {}
            for t, s in zip(texts_a, scores_a):
                sv = float(s.item())
                if t not in map_a or sv > map_a[t]:
                    map_a[t] = sv
            for t, s in zip(texts_b, scores_b):
                sv = float(s.item())
                if t not in map_b or sv > map_b[t]:
                    map_b[t] = sv

            vocab = list(set(map_a.keys()) | set(map_b.keys()))
            if not vocab:
                # In degenerate case, return uniform tiny distributions
                p = torch.full((1,), 1.0, device=device)
                q = torch.full((1,), 1.0, device=device)
                return p, q

            # Use a big negative for missing items so softmax pushes prob to ~0
            neg_inf = -1e9
            vec_a = torch.tensor(
                [map_a.get(it, neg_inf) for it in vocab], device=device
            )
            vec_b = torch.tensor(
                [map_b.get(it, neg_inf) for it in vocab], device=device
            )

            p = torch.softmax(vec_a, dim=0)
            q = torch.softmax(vec_b, dim=0)
            return p, q

        with open(out_path, "w", encoding="utf-8") as f, torch.no_grad():
            base = 0
            for batch in dl:
                batch = self._prepare_inputs(batch)
                prompt_ids = self._prompt_ids_from_batch_keys(
                    batch
                )  # e.g., [1, 2, 3, ...]
                K_prompts = len(prompt_ids)
                if K_prompts == 0:
                    continue

                B = batch[f"input_ids_{prompt_ids[0]}"].size(0)

                # For each prompt, generate top-k sequences for the whole batch
                generated: Dict[int, List[List[str]]] = {}
                beam_scores: Dict[int, torch.Tensor] = {}

                for pid in prompt_ids:
                    ids = batch[f"input_ids_{pid}"]
                    msk = batch[f"attention_mask_{pid}"]

                    gen_out = model.generate(
                        input_ids=ids,
                        attention_mask=msk,
                        max_length=max_len,
                        num_beams=num_beams,
                        num_return_sequences=num_return,
                        output_scores=True,
                        return_dict_in_generate=True,
                    )

                    # Reshape [B * num_return, L] -> [B, num_return, L]
                    seqs = gen_out.sequences.view(B, num_return, -1)
                    scores = gen_out.sequences_scores.view(
                        B, num_return
                    )  # [B, num_return]

                    # Decode beams into strings
                    pred_texts = [
                        [
                            tokenizer.decode(
                                seqs[b, k], skip_special_tokens=True
                            ).strip()
                            for k in range(num_return)
                        ]
                        for b in range(B)
                    ]

                    generated[pid] = pred_texts
                    beam_scores[pid] = scores

                # Now process each example j in batch
                for j in range(B):
                    src = dataset.samples[base + j]

                    target_text = str(src.get("target", "")).strip()
                    if not target_text:
                        continue  # skip if no gold

                    # Per example, construct ranked predictions for each prompt
                    per_prompt_order: Dict[int, List[str]] = {}
                    per_prompt_top1: Dict[int, str] = {}

                    for pid in prompt_ids:
                        preds = generated[pid][j]  # list of top-k strings

                        # Dedupe while preserving order
                        uniq = []
                        seen = set()
                        for it in preds:
                            if it not in seen:
                                seen.add(it)
                                uniq.append(it)

                        if not uniq:
                            # No valid prediction from this prompt, we skip this example
                            break

                        per_prompt_order[pid] = uniq
                        per_prompt_top1[pid] = uniq[0]

                    # If any prompt had no predictions, skip this example
                    if len(per_prompt_order) < K_prompts:
                        continue

                    # Initialize per prompt counters if needed
                    for pid in prompt_ids:
                        if pid not in per_prompt_exact:
                            per_prompt_exact[pid] = 0
                        if pid not in per_prompt_hits:
                            per_prompt_hits[pid] = {K: 0 for K in ks}
                        if pid not in per_prompt_ndcg:
                            per_prompt_ndcg[pid] = {K: 0.0 for K in ks}

                    # Per prompt metrics
                    for pid in prompt_ids:
                        order = per_prompt_order[pid]
                        try:
                            pos = order.index(target_text)
                        except ValueError:
                            pos = 10**9  # effectively "not in top-K"

                        for K in ks:
                            if pos < K:
                                per_prompt_hits[pid][K] += 1
                                per_prompt_ndcg[pid][K] += self._dcg_at_pos(pos)
                        per_prompt_exact[pid] += int(pos == 0)

                    # Cross prompt consistency
                    if K_prompts >= 2:
                        for pa, pb in itertools.combinations(prompt_ids, 2):
                            list_a = per_prompt_order[pa]
                            list_b = per_prompt_order[pb]
                            if not list_a or not list_b:
                                continue

                            pair_count += 1

                            # top1 agreement
                            top1_agree_sum += int(list_a[0] == list_b[0])

                            # Jaccard@K on item sets
                            for K in ks:
                                set_a = set(list_a[:K])
                                set_b = set(list_b[:K])
                                denom_j = max(1, len(set_a | set_b))
                                jacc_sum[K] += len(set_a & set_b) / denom_j

                            # L2 and JS on prob distributions over union of items
                            scores_a = beam_scores[pa][j]  # [num_return]
                            scores_b = beam_scores[pb][j]  # [num_return]
                            texts_a = generated[pa][j]
                            texts_b = generated[pb][j]

                            device = scores_a.device
                            p_vec, q_vec = _softmax_scores_over_union(
                                texts_a, scores_a, texts_b, scores_b, device
                            )

                            # L2
                            l2_sum += float(torch.norm(p_vec - q_vec, p=2).item())

                            # JS divergence
                            eps = 1e-8
                            p_ = torch.clamp(p_vec, min=eps)
                            q_ = torch.clamp(q_vec, min=eps)
                            m_ = 0.5 * (p_ + q_)
                            js_div = 0.5 * (
                                torch.sum(p_ * (p_.log() - m_.log()))
                                + torch.sum(q_ * (q_.log() - m_.log()))
                            )
                            js_sum += float(js_div.item())

                    # Write JSONL for this example
                    k_out = min(topk, min(len(v) for v in per_prompt_order.values()))
                    topk_by_prompt = {
                        str(pid): [
                            {
                                "item": per_prompt_order[pid][ri],
                                "score_rank": ri + 1,
                            }
                            for ri in range(k_out)
                        ]
                        for pid in prompt_ids
                    }

                    json_obj = {
                        "input_idx": src.get("input_1", None),
                        "target": target_text,
                        "top1_by_prompt": {
                            str(pid): per_prompt_top1[pid] for pid in prompt_ids
                        },
                        "ranked_items_by_prompt": {
                            str(pid): per_prompt_order[pid][:k_out]
                            for pid in prompt_ids
                        },
                        "topk_by_prompt": topk_by_prompt,
                        "gen": {
                            "max_length": int(max_len),
                            "num_beams": int(num_beams),
                            "num_return_sequences": int(num_return),
                        },
                    }
                    f.write(json.dumps(json_obj, ensure_ascii=False) + "\n")

                    total += 1
                    # better progress bar: use dataset size if available
                    try:
                        total_size = len(dataset.samples)
                    except AttributeError:
                        total_size = len(dl) * B

                    print_progress(current=total, total=total_size)

                    if num_examples and total >= int(num_examples):
                        break

                base += B
                if num_examples and total >= int(num_examples):
                    break

        # Aggregate metrics
        num_prompts = len(per_prompt_exact)
        pair_denom = max(1, pair_count)

        metrics = {
            "total": total,
            "Hit@K_by_prompt": {
                str(pid): {
                    K: (per_prompt_hits[pid][K] / total) if total else 0.0 for K in ks
                }
                for pid in per_prompt_hits
            },
            "NDCG@K_by_prompt": {
                str(pid): {
                    K: (per_prompt_ndcg[pid][K] / total) if total else 0.0 for K in ks
                }
                for pid in per_prompt_ndcg
            },
            "CONSISTENCY": {
                **{
                    f"mean_Jaccard@{K}": (
                        (jacc_sum[K] / pair_denom)
                        if (pair_count and num_prompts >= 2)
                        else 0.0
                    )
                    for K in ks
                },
                "top1_agreement_rate": (
                    (top1_agree_sum / pair_denom)
                    if (pair_count and num_prompts >= 2)
                    else 0.0
                ),
                "mean_L2_prob": (
                    (l2_sum / pair_denom) if (pair_count and num_prompts >= 2) else 0.0
                ),
                "mean_JS_div": (
                    (js_sum / pair_denom) if (pair_count and num_prompts >= 2) else 0.0
                ),
            },
        }

        # Save metrics
        jsonl_path = out_path.replace(".jsonl", "_beam_metrics.jsonl")
        with open(jsonl_path, "a", encoding="utf-8") as mf:
            mf.write(json.dumps(metrics, ensure_ascii=False) + "\n")

        print(f"✅ Wrote {metrics['total']} multiprompt predictions to {out_path}")
        for pid in metrics["Hit@K_by_prompt"]:
            for K in ks:
                print(
                    f"Hit@{K}_prompt[{pid}]  = {metrics['Hit@K_by_prompt'][pid][K]:.4f}"
                )
                print(
                    f"NDCG@{K}_prompt[{pid}] = {metrics['NDCG@K_by_prompt'][pid][K]:.4f}"
                )
        print("— Consistency (avg over prompt pairs) —")
        for K in ks:
            print(
                f"Mean Jaccard@{K}: {metrics['CONSISTENCY'][f'mean_Jaccard@{K}']:.4f}"
            )
        print(
            "Top1 agreement :", f"{metrics['CONSISTENCY']['top1_agreement_rate']:.6f}"
        )
        print("Mean L2(prob)  :", f"{metrics['CONSISTENCY']['mean_L2_prob']:.6f}")
        print("Mean JS div    :", f"{metrics['CONSISTENCY']['mean_JS_div']:.6f}")

        return metrics
