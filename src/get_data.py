import sys

sys.path.append("..")

from src.data_loader.data_loader import get_loader

from src.helper_func import print_progress

import json
import torch

import random
import numpy as np
from tqdm import tqdm
import os


def get_data(**kwargs):
    random_seed = 2025
    torch.manual_seed(random_seed)
    random.seed(random_seed)
    np.random.seed(random_seed)

    # sequential
    test_task_list = {"sequential": ["2-1-3"]}
    sample_numbers = {
        "rating": 0,
        "sequential": [1, 0, 0],  # equal to number of prompt templates in the task
        "explanation": 0,
        "review": 0,
        "traditional": [0, 0],
    }
    test_sample_numbers = {
        key: tuple(val) if isinstance(val, list) else val
        for key, val in sample_numbers.items()
    }
    print("test_sample_numbers", test_sample_numbers)

    print(kwargs)

    mp_dir = f"mp_data/{kwargs['dataset']}/"
    os.makedirs(mp_dir, exist_ok=True)
    for mode in ["train", "test", "val"]:
        prompt_id_list = None
        if mode == "train":
            prompt_id_list = [1, 2, 3, 4, 5, 6, 7]
        else:
            prompt_id_list = [1, 2, 3, 4, 5, 6, 7, 8, 9]
        if prompt_id_list is None:
            raise ValueError("prompt_id_list is None. Please provide a valid mode.")
        zeroshot_test_loader = get_loader(
            test_task_list,
            sample_numbers=test_sample_numbers,
            split=kwargs["dataset"],
            mode=mode,
            batch_size=1,
            workers=4,
            distributed=False,
        )
        mp_filename = f"{mp_dir}{mode}.jsonl"
        # If file exists, delete it
        if os.path.exists(mp_filename):
            os.remove(mp_filename)
        for example_index, batch in tqdm(enumerate(zeroshot_test_loader)):
            mp_sample = {}

            for prompt_id in prompt_id_list:
                if f"input_{prompt_id}" not in batch:
                    raise ValueError(f"input_{prompt_id} not in batch.")
                mp_sample[f"input_{prompt_id}"] = batch[f"input_{prompt_id}"][0]
                mp_sample[f"item_spans_{prompt_id}"] = batch[f"spans_{prompt_id}"][0]
            mp_sample["target"] = batch["target_text"][0]
            mp_sample["candidates"] = batch["candidates"][0].split(" , ")
            
            with open(
                mp_filename,
                "a",
                encoding="utf-8",
            ) as f:
                f.write(json.dumps(mp_sample, ensure_ascii=False) + "\n")
            print_progress(example_index + 1, len(zeroshot_test_loader))
