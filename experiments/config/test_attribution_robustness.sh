#!/bin/bash
echo "Starting to calculate attribution score..."
python ../../src/test_attr_cons.py \
  --model_name makitanikaze/P5_toys_small \
  --test_file ../../mp_data/toys/test.jsonl \
  --output_dir ../../experiments/output_small/nolayer/rar_toys_01 \
  --generation_num_beams 20 \
  --attr_metrics_k 1,2,3,5,10 \
  --attr_topk 10 \
  --attr_rank_corr kendall \
  --test_prompt_id_list "[1, 2, 3, 4, 5, 6, 7, 8, 9]"
