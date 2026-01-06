#!/bin/bash
echo "Starting training..."

python ../../train_model.py \
  --model_name makitanikaze/P5_toys_small \
  --train_file ../../mp_data/toys/train.jsonl \
  --val_file ../../mp_data/toys/val.jsonl \
  --test_file ../../mp_data/toys/test.jsonl \
  --output_dir ../output/PIAR/toys \
  --do_train \
  --learning_rate 3e-5 \
  --num_train_epochs 4 \
  --per_device_train_batch_size 16 \
  --per_device_eval_batch_size 32 \
  --lambda_cons_attr 0.1 \
  --attr_metric l1 \
  --fp16 \
  --auto_resume \
  --topk 10 \
  --generation_max_length 16 \
  --generation_num_beams 20 \
  --load_trained_if_exists \
  --max_input_length 256 \
  --attr_center mince \
  --check_gradient \
  --num_epoch_warmup 0 \
  --metrics_k 1,5,10 \
  --seed 2025 \
  --log_per_sample \
  --sum_attr_per_item \
  --save_test_prompt_pred \
  --test_prompt_id_list "[1, 2, 3, 4, 5, 6, 7, 8, 9]"
