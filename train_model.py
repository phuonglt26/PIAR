import argparse, json, os, inspect

from transformers import (
    AutoTokenizer,
    AutoModelForSeq2SeqLM,
    Seq2SeqTrainer,
    Seq2SeqTrainingArguments,
)
from transformers.trainer_utils import get_last_checkpoint
from src.model import P5FairAttrTrainer

# >>>> MultiPrompt Dataset/Collator we already have in prompt_loader <<<<
from src.prompt_loader import (
    MultiPromptSpanCollator,
    JsonlSeq2SeqMultiPromptSpanDataset,
)


def parse_args():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model_name", type=str, required=True)
    ap.add_argument(
        "--train_file", type=str, help="JSONL multiprompt + (optional) spans"
    )
    ap.add_argument(
        "--val_file", type=str, default="", help="JSONL multiprompt + (optional) spans"
    )
    ap.add_argument(
        "--test_file", type=str, default="", help="JSONL multiprompt (eval)"
    )
    ap.add_argument("--output_dir", type=str, default="./p5_fair_attr")

    # training
    ap.add_argument("--learning_rate", type=float, default=3e-5)
    ap.add_argument("--num_train_epochs", type=float, default=3.0)
    ap.add_argument("--per_device_train_batch_size", type=int, default=1)
    ap.add_argument("--per_device_eval_batch_size", type=int, default=1)
    ap.add_argument("--weight_decay", type=float, default=0.01)
    ap.add_argument("--warmup_ratio", type=float, default=0.0)
    ap.add_argument("--logging_steps", type=int, default=50)
    ap.add_argument("--save_total_limit", type=int, default=2)
    ap.add_argument("--fp16", action="store_true")

    # lengths
    ap.add_argument("--max_input_length", type=int, default=256)
    ap.add_argument("--max_target_length", type=int, default=16)
    ap.add_argument("--max_candidate_length", type=int, default=8)

    # generation (eval)
    ap.add_argument("--generation_max_length", type=int, default=32)
    ap.add_argument("--generation_num_beams", type=int, default=1)

    # consistency knobs
    ap.add_argument("--lambda_cons_attr", type=float, default=0.0)
    ap.add_argument(
        "--attr_metric", type=str, default="l2", choices=["l2", "l1", "mse", "jsd"]
    )
    ap.add_argument(
        "--attr_norm",
        type=str,
        default=None,
        choices=[
            "l1",
            "l2",
            "relu",
            "linf",
            "zscore",
            "tanh",
            "signed",
            "cosine",
            "sum_norm",
        ],
        help="Attribution Normalization function after computing attributions for each prompt items",
    )
    ap.add_argument(
        "--attr_center",
        type=str,
        default="mince",
        choices=["mince", "allpairs", "random", "fixed", "mean"],
        help="Pairing scheme for consistency losses",
    )

    # control
    ap.add_argument("--do_train", action="store_true")
    ap.add_argument("--load_trained_if_exists", action="store_true")
    ap.add_argument("--auto_resume", action="store_true")
    ap.add_argument("--resume_from_checkpoint", action="store_true")

    # eval outputs
    ap.add_argument(
        "--save_test_prompt_pred",
        action="store_true",
        help="Save test predictions by prompt",
    )
    ap.add_argument("--metrics_k", type=str, default="5,10")
    ap.add_argument("--topk", type=int, default=5)
    ap.add_argument("--num_test_examples", type=int, default=None)
    ap.add_argument(
        "--check_gradient", action="store_true", help="Check gradient flow."
    )
    ap.add_argument(
        "--detach_center", action="store_true", help="Use to detach center."
    )
    ap.add_argument("--num_epoch_warmup", type=int, default=None, help="Warmup epochs")
    ap.add_argument("--train_prompt_id_list", type=json.loads, default=None)
    ap.add_argument("--test_prompt_id_list", type=json.loads, default=None)
    ap.add_argument("--center_prompt_id", type=int, default=None)
    ap.add_argument(
        "--log_per_sample",
        action="store_true",
        help="if log_per_sample that means log each loss per sample",
    )
    ap.add_argument(
        "--get_eval",
        action="store_true",
        help="if get_eval that means run eval after every epoch",
    )
    ap.add_argument(
        "--sum_attr_per_item",
        action="store_true",
        help="if sum_attr_per_item that means sum tokens attribution for each item else use tokens only.",
    )
    ap.add_argument(
        "--seed",
        type=int,
        default=2025,
    )
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

    tokenizer = AutoTokenizer.from_pretrained(args.model_name, use_fast=True)
    print(f"🌟 Using model: {args.model_name}")
    if tokenizer.pad_token is None and tokenizer.eos_token is not None:
        tokenizer.pad_token = tokenizer.eos_token

    load_trained = args.load_trained_if_exists and _has_model_weights(args.output_dir)
    if load_trained:
        print(f"🔄 Found trained weights in {args.output_dir} — loading them.")
        model = AutoModelForSeq2SeqLM.from_pretrained(args.output_dir)
    else:
        print(f"🆕 Loading base model: {args.model_name}")
        model = AutoModelForSeq2SeqLM.from_pretrained(args.model_name)

    # Datasets (MultiPrompt)
    def build_dataset(path: str, prompt_id_list=None):
        return (
            JsonlSeq2SeqMultiPromptSpanDataset(
                path,
                tokenizer,
                max_input_length=args.max_input_length,
                max_target_length=args.max_target_length,
                max_candidate_length=args.max_candidate_length,
                prompt_id_list=prompt_id_list,
            )
            if path
            else None
        )

    train_ds = build_dataset(args.train_file, args.train_prompt_id_list)
    val_ds = build_dataset(args.val_file, args.train_prompt_id_list)
    test_ds = build_dataset(args.test_file, args.test_prompt_id_list)

    collator = MultiPromptSpanCollator(tokenizer)

    training_args = Seq2SeqTrainingArguments(
        output_dir=args.output_dir,
        eval_strategy="epoch" if args.get_eval else "no",
        per_device_train_batch_size=args.per_device_train_batch_size,
        per_device_eval_batch_size=args.per_device_eval_batch_size,
        learning_rate=args.learning_rate,
        weight_decay=args.weight_decay,
        num_train_epochs=args.num_train_epochs,
        save_total_limit=args.save_total_limit,
        logging_steps=args.logging_steps,
        warmup_ratio=args.warmup_ratio,
        gradient_accumulation_steps=1,
        seed=args.seed,
        fp16=args.fp16,
        report_to=["none"],
        remove_unused_columns=False,  # keep custom fields
        predict_with_generate=False,
        logging_strategy="steps",
        logging_first_step=True,
    )

    # Tokenizer vs processing_class (HF version)
    sig = inspect.signature(Seq2SeqTrainer.__init__)
    use_processing = "processing_class" in sig.parameters

    trainer_kwargs = dict(
        model=model,
        args=training_args,
        train_dataset=train_ds if args.do_train else None,
        eval_dataset=val_ds if val_ds is not None else None,
        data_collator=collator,
    )
    if use_processing:
        trainer_kwargs["processing_class"] = tokenizer
    else:
        trainer_kwargs["tokenizer"] = tokenizer

    trainer = P5FairAttrTrainer(**trainer_kwargs)

    # attach custom knobs to trainer
    trainer.lambda_cons_attr = float(args.lambda_cons_attr)
    trainer.attr_metric = str(args.attr_metric)
    trainer.attr_norm = args.attr_norm
    trainer.attr_center = str(args.attr_center)
    trainer.num_beams = args.generation_num_beams
    trainer.max_length = args.max_input_length
    trainer.center_prompt_id = args.center_prompt_id
    trainer.log_per_sample = args.log_per_sample
    trainer.sum_attr_per_item = args.sum_attr_per_item

    if args.attr_center == "fixed" and args.center_prompt_id is None:
        raise ValueError("attr_center 'fixed' requires --center_prompt_id to be set.")

    trainer.check_gradient = (
        args.check_gradient if hasattr(args, "check_gradient") else False
    )
    trainer.detach_center = (
        args.detach_center if hasattr(args, "detach_center") else False
    )
    trainer.num_epoch_warmup = (
        args.num_epoch_warmup if hasattr(args, "num_epoch_warmup") else None
    )  # warmup epochs for cons losses

    # Train / resume
    if args.do_train:
        resume_ckpt = None
        if getattr(args, "resume_from_checkpoint", ""):
            resume_ckpt = args.resume_from_checkpoint
            print(f"▶️ Resuming from explicit checkpoint: {resume_ckpt}")
        elif args.auto_resume:
            last_ckpt = (
                get_last_checkpoint(args.output_dir)
                if os.path.isdir(args.output_dir)
                else None
            )
            if last_ckpt is not None:
                resume_ckpt = last_ckpt
                print(f"▶️ Auto-resuming from last checkpoint: {resume_ckpt}")

        trainer.train(resume_from_checkpoint=resume_ckpt)
        trainer.save_model(args.output_dir)
        tokenizer.save_pretrained(args.output_dir)
    else:
        print("⏭️ Skipping training per --do_train False.")

    # Optional: evaluation save
    if args.save_test_prompt_pred and test_ds is not None:
        ks = [int(k) for k in str(args.metrics_k).split(",") if k.strip().isdigit()]
        out_path = os.path.join(args.output_dir, "test_predictions_multiprompt.jsonl")
        trainer.data_collator = MultiPromptSpanCollator(
            tokenizer, args.test_prompt_id_list
        )
        trainer.generate_prediction_beam(
            test_ds,
            tokenizer,
            out_path=out_path,
            generation_max_length=args.generation_max_length,
            generation_num_beams=20,
            topk=args.topk,
            ks=ks,
            num_examples=args.num_test_examples,
        )


if __name__ == "__main__":
    main()
