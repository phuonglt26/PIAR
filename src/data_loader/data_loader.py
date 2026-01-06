import numpy as np
import json, gzip, random, pickle, torch, os

from torch.utils.data.distributed import DistributedSampler
from torch.utils.data import DataLoader, Dataset
from collections import defaultdict


def load_json(file_path):
    with open(file_path, "r") as f:
        return json.load(f)


def load_pickle(filename):
    with open(filename, "rb") as f:
        return pickle.load(f)


def ReadLineFromFile(path):
    lines = []
    with open(path, "r") as fd:
        for line in fd:
            lines.append(line.rstrip("\n"))
    return lines


def parse(path):
    g = gzip.open(path, "r")
    for l in g:
        yield eval(l)


class P5_Amazon_Dataset(Dataset):
    def __init__(
        self,
        all_tasks,
        task_list,
        sample_numbers,
        mode="train",
        split="toys",
        rating_augment=False,
        sample_type="random",
    ):
        self.all_tasks = all_tasks
        self.task_list = task_list
        self.sample_numbers = sample_numbers
        self.split = split
        self.rating_augment = rating_augment
        self.sample_type = sample_type
        self.purchase_history = None

        print("Data sources: ", split.split(","))
        self.mode = mode
        if self.mode == "train":
            self.review_data = load_pickle(
                os.path.join("data", split, "review_splits.pkl")
            )["train"]
            self.exp_data = load_pickle(os.path.join("data", split, "exp_splits.pkl"))[
                "train"
            ]
            if self.rating_augment:
                self.rating_data = load_pickle(
                    os.path.join("data", split, "rating_splits_augmented.pkl")
                )["train"]
            else:
                self.rating_data = self.review_data
        elif self.mode == "val":
            self.review_data = load_pickle(
                os.path.join("data", split, "review_splits.pkl")
            )["val"]
            self.exp_data = load_pickle(os.path.join("data", split, "exp_splits.pkl"))[
                "val"
            ]
            if self.rating_augment:
                self.rating_data = load_pickle(
                    os.path.join("data", split, "rating_splits_augmented.pkl")
                )["val"]
            else:
                self.rating_data = self.review_data
        elif self.mode == "test":
            self.review_data = load_pickle(
                os.path.join("data", split, "review_splits.pkl")
            )["test"]
            self.exp_data = load_pickle(os.path.join("data", split, "exp_splits.pkl"))[
                "test"
            ]
            if self.rating_augment:
                self.rating_data = load_pickle(
                    os.path.join("data", split, "rating_splits_augmented.pkl")
                )["test"]
            else:
                self.rating_data = self.review_data
            self.zeroshot_exp_data = load_pickle(
                os.path.join("data", "beauty", "zeroshot_exp_splits.pkl")
            )  # change to dataset to be transferred (e.g., 'beauty')
        else:
            raise NotImplementedError

        self.sequential_data = ReadLineFromFile(
            os.path.join("data", split, "sequential_data.txt")
        )
        item_count = defaultdict(int)
        user_items = defaultdict()

        for line in self.sequential_data:
            user, items = line.strip().split(" ", 1)
            items = items.split(" ")
            items = [int(item) for item in items]
            user_items[user] = items
            for item in items:
                item_count[item] += 1

        self.all_item = list(item_count.keys())
        count = list(item_count.values())
        sum_value = np.sum([x for x in count])
        self.probability = [value / sum_value for value in count]
        self.user_items = user_items

        if self.mode == "test":
            self.negative_samples = ReadLineFromFile(
                os.path.join("data", split, "negative_samples.txt")
            )

        datamaps = load_json(os.path.join("data", split, "datamaps.json"))
        self.user2id = datamaps["user2id"]
        self.item2id = datamaps["item2id"]
        self.user_list = list(datamaps["user2id"].keys())
        self.item_list = list(datamaps["item2id"].keys())
        self.id2item = datamaps["id2item"]

        self.user_id2name = load_pickle(os.path.join("data", split, "user_id2name.pkl"))

        self.meta_data = []
        for meta in parse(os.path.join("data", split, "meta.json.gz")):
            self.meta_data.append(meta)
        self.meta_dict = {}
        for i, meta_item in enumerate(self.meta_data):
            self.meta_dict[meta_item["asin"]] = i

        print("compute_datum_info")
        self.total_length = 0
        self.datum_info = []
        self.compute_datum_info()

    # compute_datum_info function intends to plan which data sample to be used for which task group according to the sample numbers in train_sample_numbers of pretrain.py
    def compute_datum_info(self):
        curr = 0
        for key in list(self.task_list.keys()):
            if key == "rating":
                self.total_length += len(self.rating_data) * self.sample_numbers[key]
                for i in range(self.total_length - curr):
                    self.datum_info.append(
                        (i + curr, key, i // self.sample_numbers[key])
                    )
                curr = self.total_length
            elif key == "sequential":
                # The first group of sequential prompts (directly predict next item): 2-1 to 2-6 and 2-13 to 2-19
                if sum(
                    [
                        0 < int(ind.split("-")[1]) <= 6
                        or 12 < int(ind.split("-")[1]) <= 19
                        for ind in self.task_list[key]
                    ]
                ):
                    self.total_length += (
                        len(self.sequential_data) * self.sample_numbers[key][0]
                    )
                    for i in range(self.total_length - curr):
                        self.datum_info.append(
                            (i + curr, key, i // self.sample_numbers[key][0])
                        )
                    curr = self.total_length
                # The second group of sequential prompts (predict next item from a candidate list): 2-7 to 2-10
                if sum(
                    [6 < int(ind.split("-")[1]) <= 10 for ind in self.task_list[key]]
                ):
                    self.total_length += (
                        len(self.sequential_data) * self.sample_numbers[key][1]
                    )
                    for i in range(self.total_length - curr):
                        self.datum_info.append(
                            (i + curr, key, i // self.sample_numbers[key][1])
                        )
                    curr = self.total_length
                # The third group of sequential prompts (predict yes or no for each user-item pair): 2-11 to 2-12
                if sum(
                    [10 < int(ind.split("-")[1]) <= 12 for ind in self.task_list[key]]
                ):
                    self.total_length += (
                        len(self.sequential_data) * self.sample_numbers[key][2]
                    )
                    for i in range(self.total_length - curr):
                        self.datum_info.append(
                            (i + curr, key, i // self.sample_numbers[key][2])
                        )

                    # for ind in self.task_list[key]:
                    #     self.total_length += len(self.sequential_data)
                    #     for i in range(len(self.sequential_data)):
                    #         self.datum_info.append(
                    #             (
                    #                 self.total_length - len(self.sequential_data) + i,
                    #                 key,
                    #                 i,
                    #                 ind,
                    #             )
                    #         )  # thêm template_id
                    curr = self.total_length
            elif key == "explanation":
                self.total_length += len(self.exp_data) * self.sample_numbers[key]
                for i in range(self.total_length - curr):
                    self.datum_info.append(
                        (i + curr, key, i // self.sample_numbers[key])
                    )
                curr = self.total_length
            elif key == "review":
                self.total_length += len(self.review_data) * self.sample_numbers[key]
                for i in range(self.total_length - curr):
                    self.datum_info.append(
                        (i + curr, key, i // self.sample_numbers[key])
                    )
                curr = self.total_length
            elif key == "traditional":
                # The first group of direct recommendation prompts (predict yes or no for each user-item pair): 5-1 to 5-4
                if sum(
                    [0 < int(ind.split("-")[1]) <= 4 for ind in self.task_list[key]]
                ):
                    self.total_length += len(self.user2id) * self.sample_numbers[key][0]
                    for i in range(self.total_length - curr):
                        self.datum_info.append(
                            (i + curr, key, i // self.sample_numbers[key][0])
                        )
                    curr = self.total_length
                # The second group of direct recommendation prompts (choose one item from 100 candidates): 5-5 to 5-8
                if sum(
                    [4 < int(ind.split("-")[1]) <= 8 for ind in self.task_list[key]]
                ):
                    self.total_length += len(self.user2id) * self.sample_numbers[key][1]
                    for i in range(self.total_length - curr):
                        self.datum_info.append(
                            (i + curr, key, i // self.sample_numbers[key][1])
                        )
                    curr = self.total_length
            elif key == "zeroshot":
                if sum(
                    [0 < int(ind.split("-")[1]) <= 7 for ind in self.task_list[key]]
                ):
                    self.total_length += (
                        len(self.zeroshot_exp_data) * self.sample_numbers[key][0]
                    )
                    for i in range(self.total_length - curr):
                        self.datum_info.append(
                            (i + curr, key, i // self.sample_numbers[key][0])
                        )
                    curr = self.total_length
            else:
                raise NotImplementedError

    # use Gaussian sampling to augment rating scores
    def gaussian_sampling(self, datum):
        if self.mode == "train":
            if int(datum["overall"]) == 1:
                sampled_rating = round(
                    torch.normal(
                        mean=torch.tensor((1.0 + 1.4) / 2),
                        std=torch.tensor((1.4 - 1.0) / 4),
                    ).item(),
                    1,
                )
            elif int(datum["overall"]) == 2:
                sampled_rating = round(
                    torch.normal(
                        mean=torch.tensor((1.5 + 2.4) / 2),
                        std=torch.tensor((2.4 - 1.5) / 4),
                    ).item(),
                    1,
                )
            elif int(datum["overall"]) == 3:
                sampled_rating = round(
                    torch.normal(
                        mean=torch.tensor((2.5 + 3.4) / 2),
                        std=torch.tensor((3.4 - 2.5) / 4),
                    ).item(),
                    1,
                )
            elif int(datum["overall"]) == 4:
                sampled_rating = round(
                    torch.normal(
                        mean=torch.tensor((3.5 + 4.4) / 2),
                        std=torch.tensor((4.4 - 3.5) / 4),
                    ).item(),
                    1,
                )
            else:
                sampled_rating = round(
                    torch.normal(
                        mean=torch.tensor((4.5 + 5.0) / 2),
                        std=torch.tensor((5.0 - 4.5) / 4),
                    ).item(),
                    1,
                )
            if sampled_rating > 5.0:
                sampled_rating = 5.0
            if sampled_rating < 1.0:
                sampled_rating = 1.0
            return str(sampled_rating)
        else:
            return int(datum["overall"])

    @staticmethod
    def _build_input_m_and_char_spans(source, user_id, purchase_history, sep=" , "):
        """
        Format input_m and compute char spans for each item in purchase_history
        exactly as they appear inside input_m.

        Args:
            task_template: dict with key "input_m" (a Python .format template)
                        e.g. {"input_m": "User {} | History: {} | Query: ..."}
            user_id:       user id to inject into the template
            purchase_history: list[str] or list[int]; items will be cast to str
            sep:           separator used when joining history (default: " , ")

        Returns:
            input_m: str — the fully formatted prompt
            spans:   list[list[int, int]] — char spans [start, end) for each item
                    in the same order as purchase_history
        """
        items = [str(x) for x in purchase_history]
        history_str = sep.join(items)
        text_input = source.format(user_id, history_str)

        spans = []
        # Prefer the fast, exact method: locate the whole history block, then compute offsets
        block_start = text_input.find(history_str)
        if block_start != -1:
            cur = block_start
            for i, it in enumerate(items):
                spans.append([cur, cur + len(it)])
                cur += len(it)
                if i != len(items) - 1:
                    cur += len(sep)
            return text_input, spans

        # Fallback: sequential search (in case the template alters the history formatting)
        search_from = 0
        for it in items:
            pos = text_input.find(it, search_from)
            if pos == -1:
                raise ValueError(
                    f"Could not find item '{it}' in the formatted input_m."
                )
            spans.append([pos, pos + len(it)])
            search_from = pos + len(it)

        return text_input, spans

    def __len__(self):
        return self.total_length

    def __getitem__(self, idx):

        out_dict = {}

        loss_weight = 1.0

        datum_info_idx = self.datum_info[idx]
        assert datum_info_idx[0] == idx
        if len(datum_info_idx) == 3:
            task_name = datum_info_idx[1]
            datum_idx = datum_info_idx[2]
        elif len(datum_info_idx) == 4:
            task_name = datum_info_idx[1]
            datum_idx = datum_info_idx[2]
            task_idx = datum_info_idx[3]
        else:
            raise NotImplementedError

        if task_name == "sequential":
            sequential_datum = self.sequential_data[datum_idx]
            sequence = sequential_datum.split()
            user_id = sequence[0]
            user_desc = self.user_id2name[user_id]
            if self.mode == "train":
                end_candidates = [
                    _ for _ in range(max(2, len(sequence) - 6), len(sequence) - 3)
                ]
                end_index = random.randint(0, len(end_candidates) - 1)
                end_pos = end_candidates[end_index]
                start_candidates = [_ for _ in range(1, min(4, end_pos))]
                start_index = random.randint(0, len(start_candidates) - 1)
                start_pos = start_candidates[start_index]
                purchase_history = sequence[
                    start_pos : end_pos + 1
                ]  # sample a history sequence from the full user purchase history
                target_item = sequence[end_pos + 1]
            elif self.mode == "val":
                purchase_history = sequence[1:-2]
                target_item = sequence[-2]
            elif self.mode == "test":
                purchase_history = sequence[1:-1]
                target_item = sequence[-1]
            else:
                raise NotImplementedError
            self.purchase_history = purchase_history
            task_candidates = self.task_list[task_name]

            task_idx = datum_info_idx[0] % len(task_candidates)
            task_template = self.all_tasks["sequential"][task_candidates[task_idx]]
            assert task_template["task"] == "sequential"
            # print(task_template["id"])
            if task_template["id"] in [
                "2-1-3",
            ]:
                self.source_texts = {}
                self.item_spans = {}
                rand_prob = 1  # random.random()
                if rand_prob > 0.5:
                    prompt_id = 1
                    while f"input_{prompt_id}" in task_template:
                        source_text, item_spans = self._build_input_m_and_char_spans(
                            task_template[f"input_{prompt_id}"],
                            user_id,
                            purchase_history,
                            sep=" , ",
                        )
                        self.source_texts[f"source_text_{prompt_id}"] = source_text
                        self.item_spans[f"item_spans_{prompt_id}"] = item_spans
                        prompt_id += 1

                else:
                    prompt_id = 1
                    while f"input_{prompt_id}" in task_template:
                        source_text, item_spans = self._build_input_m_and_char_spans(
                            task_template[f"input_{prompt_id}"],
                            user_id,
                            purchase_history,
                            sep=" -> ",
                        )
                        self.source_texts[f"source_text_{prompt_id}"] = source_text
                        self.item_spans[f"item_spans_{prompt_id}"] = item_spans
                        prompt_id += 1

                target_text = task_template["target"].format(target_item)
                if self.mode in ["train", "val"]:
                    user_seq = self.user_items[user_id]
                    candidate_samples = []
                    candidate_num = 99
                    while len(candidate_samples) < candidate_num:
                        if self.sample_type == "random":
                            sample_ids = np.random.choice(
                                self.all_item, candidate_num, replace=False
                            )
                        else:
                            sample_ids = np.random.choice(
                                self.all_item,
                                candidate_num,
                                replace=False,
                                p=self.probability,
                            )
                        sample_ids = [
                            str(item)
                            for item in sample_ids
                            if item not in user_seq and item not in candidate_samples
                        ]
                        candidate_samples.extend(sample_ids)
                    candidate_samples = candidate_samples[:candidate_num]
                elif self.mode == "test":
                    assert (
                        user_id
                        == self.negative_samples[int(user_id) - 1].split(" ", 1)[0]
                    )
                    candidate_samples = (
                        self.negative_samples[int(user_id) - 1]
                        .split(" ", 1)[1]
                        .split(" ")
                    )
                else:
                    raise NotImplementedError
                candidate_samples.extend([target_item])
                random.shuffle(candidate_samples)
                candidates = task_template["candidates"].format(
                    " , ".join(candidate_samples)
                )

        else:
            raise NotImplementedError

        if self.purchase_history is None:
            raise (
                "Purchase history is not provided. The item_ids field will not be populated."
            )
        if task_template["id"] == "2-1-3":
            prompt_id = 1

            while f"source_text_{prompt_id}" in self.source_texts:
                out_dict[f"input_{prompt_id}"] = self.source_texts[
                    f"source_text_{prompt_id}"
                ]
                out_dict[f"spans_{prompt_id}"] = self.item_spans[
                    f"item_spans_{prompt_id}"
                ]
                prompt_id += 1
            out_dict["target_text"] = target_text
            out_dict["candidates"] = candidates

            out_dict["task"] = task_template["task"]
        self.task_template_id = task_template["id"]
        return out_dict

    def collate_fn(self, batch):
        batch_entry = {}

        B = len(batch)
        if self.task_template_id in ["2-1-3"]:
            tasks = []
            candidates = []
            target_text = []
            source_texts = {}
            item_spans = {}

            for i, entry in enumerate(batch):
                prompt_id = 1
                while f"input_{prompt_id}" in entry:
                    if f"input_{prompt_id}" not in source_texts:
                        source_texts[f"input_{prompt_id}"] = []
                        item_spans[f"spans_{prompt_id}"] = []
                    source_texts[f"input_{prompt_id}"].append(
                        entry[f"input_{prompt_id}"]
                    )
                    item_spans[f"spans_{prompt_id}"].append(entry[f"spans_{prompt_id}"])
                    prompt_id += 1

                if "task" in entry:
                    tasks.append(entry["task"])

                if "target_text" in entry:
                    target_text.append(entry["target_text"])
                if "candidates" in entry:
                    candidates.append(entry["candidates"])

            batch_entry["task"] = tasks

            prompt_id = 1
            while f"input_{prompt_id}" in source_texts:
                batch_entry[f"input_{prompt_id}"] = source_texts[f"input_{prompt_id}"]
                batch_entry[f"spans_{prompt_id}"] = item_spans[f"spans_{prompt_id}"]
                prompt_id += 1

            batch_entry["target_text"] = target_text
            batch_entry["candidates"] = candidates

        return batch_entry


class P5_Yelp_Dataset(Dataset):
    def __init__(
        self,
        all_tasks,
        task_list,
        sample_numbers,
        mode="train",
        split="toys",
        rating_augment=False,
        sample_type="random",
    ):
        self.all_tasks = all_tasks
        self.task_list = task_list
        self.sample_numbers = sample_numbers
        self.split = split
        self.rating_augment = rating_augment
        self.sample_type = sample_type
        self.purchase_history = None

        print("Data sources: ", split.split(","))
        self.mode = mode
        if self.mode == "train":
            self.review_data = load_pickle(
                os.path.join("data", split, "review_splits.pkl")
            )["train"]
            self.exp_data = load_pickle(os.path.join("data", split, "exp_splits.pkl"))[
                "train"
            ]
            if self.rating_augment:
                self.rating_data = load_pickle(
                    os.path.join("data", split, "rating_splits_augmented.pkl")
                )["train"]
            else:
                self.rating_data = self.review_data
        elif self.mode == "val":
            self.review_data = load_pickle(
                os.path.join("data", split, "review_splits.pkl")
            )["val"]
            self.exp_data = load_pickle(os.path.join("data", split, "exp_splits.pkl"))[
                "val"
            ]
            if self.rating_augment:
                self.rating_data = load_pickle(
                    os.path.join("data", split, "rating_splits_augmented.pkl")
                )["val"]
            else:
                self.rating_data = self.review_data
        elif self.mode == "test":
            self.review_data = load_pickle(
                os.path.join("data", split, "review_splits.pkl")
            )["test"]
            self.exp_data = load_pickle(os.path.join("data", split, "exp_splits.pkl"))[
                "test"
            ]
            if self.rating_augment:
                self.rating_data = load_pickle(
                    os.path.join("data", split, "rating_splits_augmented.pkl")
                )["test"]
            else:
                self.rating_data = self.review_data
        else:
            raise NotImplementedError

        self.sequential_data = ReadLineFromFile(
            os.path.join("data", split, "sequential_data.txt")
        )
        item_count = defaultdict(int)
        user_items = defaultdict()

        for line in self.sequential_data:
            user, items = line.strip().split(" ", 1)
            items = items.split(" ")
            items = [int(item) for item in items]
            user_items[user] = items
            for item in items:
                item_count[item] += 1

        self.all_item = list(item_count.keys())
        count = list(item_count.values())
        sum_value = np.sum([x for x in count])
        self.probability = [value / sum_value for value in count]
        self.user_items = user_items

        if self.mode == "test":
            self.negative_samples = ReadLineFromFile(
                os.path.join("data", split, "negative_samples.txt")
            )

        datamaps = load_json(os.path.join("data", split, "datamaps.json"))
        self.user2id = datamaps["user2id"]
        self.item2id = datamaps["item2id"]
        self.user_list = list(datamaps["user2id"].keys())
        self.item_list = list(datamaps["item2id"].keys())
        self.id2item = datamaps["id2item"]

        self.user_id2name = load_pickle(os.path.join("data", split, "user_id2name.pkl"))

        self.meta_data = load_pickle(os.path.join("data", split, "meta_data.pkl"))
        self.user_data = load_pickle(os.path.join("data", split, "user_data.pkl"))
        self.meta_dict = {}
        for i, meta_item in enumerate(self.meta_data):
            self.meta_dict[meta_item["business_id"]] = i
        self.user_meta_dict = {}
        for j, user_meta_item in enumerate(self.user_data):
            self.user_meta_dict[user_meta_item["user_id"]] = j

        print("compute_datum_info")
        self.total_length = 0
        self.datum_info = []
        self.compute_datum_info()

    def compute_datum_info(self):
        curr = 0
        for key in list(self.task_list.keys()):
            if key == "sequential":
                if sum(
                    [
                        0 < int(ind.split("-")[1]) <= 6 or int(ind.split("-")[1]) == 13
                        for ind in self.task_list[key]
                    ]
                ):
                    self.total_length += (
                        len(self.sequential_data) * self.sample_numbers[key][0]
                    )
                    for i in range(self.total_length - curr):
                        self.datum_info.append(
                            (i + curr, key, i // self.sample_numbers[key][0])
                        )
                    # for ind in self.task_list[key]:
                    #     self.total_length += len(self.sequential_data)
                    #     for i in range(len(self.sequential_data)):
                    #         self.datum_info.append(
                    #             (
                    #                 self.total_length - len(self.sequential_data) + i,
                    #                 key,
                    #                 i,
                    #                 ind,
                    #             )
                    #         )  # thêm template_id

                    curr = self.total_length
                if sum(
                    [6 < int(ind.split("-")[1]) <= 10 for ind in self.task_list[key]]
                ):
                    self.total_length += (
                        len(self.sequential_data) * self.sample_numbers[key][1]
                    )
                    for i in range(self.total_length - curr):
                        self.datum_info.append(
                            (i + curr, key, i // self.sample_numbers[key][1])
                        )
                    curr = self.total_length
                if sum(
                    [10 < int(ind.split("-")[1]) <= 12 for ind in self.task_list[key]]
                ):
                    self.total_length += (
                        len(self.sequential_data) * self.sample_numbers[key][2]
                    )
                    for i in range(self.total_length - curr):
                        self.datum_info.append(
                            (i + curr, key, i // self.sample_numbers[key][2])
                        )
                    curr = self.total_length
            else:
                raise NotImplementedError

    def gaussian_sampling(self, datum):
        if self.mode == "train":
            if int(datum["overall"]) == 1:
                sampled_rating = round(
                    torch.normal(
                        mean=torch.tensor((1.0 + 1.4) / 2),
                        std=torch.tensor((1.4 - 1.0) / 4),
                    ).item(),
                    1,
                )
            elif int(datum["overall"]) == 2:
                sampled_rating = round(
                    torch.normal(
                        mean=torch.tensor((1.5 + 2.4) / 2),
                        std=torch.tensor((2.4 - 1.5) / 4),
                    ).item(),
                    1,
                )
            elif int(datum["overall"]) == 3:
                sampled_rating = round(
                    torch.normal(
                        mean=torch.tensor((2.5 + 3.4) / 2),
                        std=torch.tensor((3.4 - 2.5) / 4),
                    ).item(),
                    1,
                )
            elif int(datum["overall"]) == 4:
                sampled_rating = round(
                    torch.normal(
                        mean=torch.tensor((3.5 + 4.4) / 2),
                        std=torch.tensor((4.4 - 3.5) / 4),
                    ).item(),
                    1,
                )
            else:
                sampled_rating = round(
                    torch.normal(
                        mean=torch.tensor((4.5 + 5.0) / 2),
                        std=torch.tensor((5.0 - 4.5) / 4),
                    ).item(),
                    1,
                )
            if sampled_rating > 5.0:
                sampled_rating = 5.0
            if sampled_rating < 1.0:
                sampled_rating = 1.0
            return str(sampled_rating)
        else:
            return int(datum["overall"])

    @staticmethod
    def _build_input_m_and_char_spans(source, user_id, purchase_history, sep=" , "):
        """
        Format input_m and compute char spans for each item in purchase_history
        exactly as they appear inside input_m.

        Args:
            task_template: dict with key "input_m" (a Python .format template)
                        e.g. {"input_m": "User {} | History: {} | Query: ..."}
            user_id:       user id to inject into the template
            purchase_history: list[str] or list[int]; items will be cast to str
            sep:           separator used when joining history (default: " , ")

        Returns:
            input_m: str — the fully formatted prompt
            spans:   list[list[int, int]] — char spans [start, end) for each item
                    in the same order as purchase_history
        """
        items = [str(x) for x in purchase_history]
        history_str = sep.join(items)
        text_input = source.format(user_id, history_str)

        spans = []
        # Prefer the fast, exact method: locate the whole history block, then compute offsets
        block_start = text_input.find(history_str)
        if block_start != -1:
            cur = block_start
            for i, it in enumerate(items):
                spans.append([cur, cur + len(it)])
                cur += len(it)
                if i != len(items) - 1:
                    cur += len(sep)
            return text_input, spans

        # Fallback: sequential search (in case the template alters the history formatting)
        search_from = 0
        for it in items:
            pos = text_input.find(it, search_from)
            if pos == -1:
                raise ValueError(
                    f"Could not find item '{it}' in the formatted input_m."
                )
            spans.append([pos, pos + len(it)])
            search_from = pos + len(it)

        return text_input, spans

    def __len__(self):
        return self.total_length

    def __getitem__(self, idx):

        out_dict = {}

        loss_weight = 1.0

        datum_info_idx = self.datum_info[idx]
        assert datum_info_idx[0] == idx
        if len(datum_info_idx) == 3:
            task_name = datum_info_idx[1]
            datum_idx = datum_info_idx[2]
        elif len(datum_info_idx) == 4:
            task_name = datum_info_idx[1]
            datum_idx = datum_info_idx[2]
            task_idx = datum_info_idx[3]
        else:
            raise NotImplementedError

        if task_name == "sequential":
            sequential_datum = self.sequential_data[datum_idx]
            sequence = sequential_datum.split()
            user_id = sequence[0]
            user_desc = self.user_id2name[user_id]
            if self.mode == "train":
                end_candidates = [
                    _ for _ in range(max(2, len(sequence) - 6), len(sequence) - 3)
                ]
                end_index = random.randint(0, len(end_candidates) - 1)
                end_pos = end_candidates[end_index]
                start_candidates = [_ for _ in range(1, min(4, end_pos))]
                start_index = random.randint(0, len(start_candidates) - 1)
                start_pos = start_candidates[start_index]
                purchase_history = sequence[
                    start_pos : end_pos + 1
                ]  # sample a history sequence from the full user purchase history
                target_item = sequence[end_pos + 1]
            elif self.mode == "val":
                purchase_history = sequence[1:-2]
                target_item = sequence[-2]
            elif self.mode == "test":
                purchase_history = sequence[1:-1]
                target_item = sequence[-1]
            else:
                raise NotImplementedError
            self.purchase_history = purchase_history
            task_candidates = self.task_list[task_name]

            task_idx = datum_info_idx[0] % len(task_candidates)
            task_template = self.all_tasks["sequential"][task_candidates[task_idx]]
            assert task_template["task"] == "sequential"
            # print(task_template["id"])
            if task_template["id"] in [
                "2-1-3",
            ]:
                self.source_texts = {}
                self.item_spans = {}
                rand_prob = 1  # random.random()
                if rand_prob > 0.5:
                    prompt_id = 1
                    while f"input_{prompt_id}" in task_template:
                        source_text, item_spans = self._build_input_m_and_char_spans(
                            task_template[f"input_{prompt_id}"],
                            user_id,
                            purchase_history,
                            sep=" , ",
                        )
                        self.source_texts[f"source_text_{prompt_id}"] = source_text
                        self.item_spans[f"item_spans_{prompt_id}"] = item_spans
                        prompt_id += 1

                else:
                    prompt_id = 1
                    while f"input_{prompt_id}" in task_template:
                        source_text, item_spans = self._build_input_m_and_char_spans(
                            task_template[f"input_{prompt_id}"],
                            user_id,
                            purchase_history,
                            sep=" -> ",
                        )
                        self.source_texts[f"source_text_{prompt_id}"] = source_text
                        self.item_spans[f"item_spans_{prompt_id}"] = item_spans
                        prompt_id += 1

                target_text = task_template["target"].format(target_item)
                if self.mode in ["train", "val"]:
                    user_seq = self.user_items[user_id]
                    candidate_samples = []
                    candidate_num = 99
                    while len(candidate_samples) < candidate_num:
                        if self.sample_type == "random":
                            sample_ids = np.random.choice(
                                self.all_item, candidate_num, replace=False
                            )
                        else:
                            sample_ids = np.random.choice(
                                self.all_item,
                                candidate_num,
                                replace=False,
                                p=self.probability,
                            )
                        sample_ids = [
                            str(item)
                            for item in sample_ids
                            if item not in user_seq and item not in candidate_samples
                        ]
                        candidate_samples.extend(sample_ids)
                    candidate_samples = candidate_samples[:candidate_num]
                elif self.mode == "test":
                    assert (
                        user_id
                        == self.negative_samples[int(user_id) - 1].split(" ", 1)[0]
                    )
                    candidate_samples = (
                        self.negative_samples[int(user_id) - 1]
                        .split(" ", 1)[1]
                        .split(" ")
                    )
                else:
                    raise NotImplementedError
                candidate_samples.extend([target_item])
                random.shuffle(candidate_samples)
                candidates = task_template["candidates"].format(
                    " , ".join(candidate_samples)
                )

        else:
            raise NotImplementedError

        if self.purchase_history is None:
            raise (
                "Purchase history is not provided. The item_ids field will not be populated."
            )
        if task_template["id"] == "2-1-3":
            prompt_id = 1

            while f"source_text_{prompt_id}" in self.source_texts:
                out_dict[f"input_{prompt_id}"] = self.source_texts[
                    f"source_text_{prompt_id}"
                ]
                out_dict[f"spans_{prompt_id}"] = self.item_spans[
                    f"item_spans_{prompt_id}"
                ]
                prompt_id += 1
            out_dict["target_text"] = target_text
            out_dict["candidates"] = candidates

            out_dict["task"] = task_template["task"]
        self.task_template_id = task_template["id"]
        return out_dict

    def collate_fn(self, batch):
        batch_entry = {}

        B = len(batch)
        if self.task_template_id in ["2-1-3"]:
            tasks = []
            candidates = []
            target_text = []
            source_texts = {}
            item_spans = {}

            for i, entry in enumerate(batch):
                prompt_id = 1
                while f"input_{prompt_id}" in entry:
                    if f"input_{prompt_id}" not in source_texts:
                        source_texts[f"input_{prompt_id}"] = []
                        item_spans[f"spans_{prompt_id}"] = []
                    source_texts[f"input_{prompt_id}"].append(
                        entry[f"input_{prompt_id}"]
                    )
                    item_spans[f"spans_{prompt_id}"].append(entry[f"spans_{prompt_id}"])
                    prompt_id += 1

                if "task" in entry:
                    tasks.append(entry["task"])

                if "target_text" in entry:
                    target_text.append(entry["target_text"])
                if "candidates" in entry:
                    candidates.append(entry["candidates"])

            batch_entry["task"] = tasks

            prompt_id = 1
            while f"input_{prompt_id}" in source_texts:
                batch_entry[f"input_{prompt_id}"] = source_texts[f"input_{prompt_id}"]
                batch_entry[f"spans_{prompt_id}"] = item_spans[f"spans_{prompt_id}"]
                prompt_id += 1

            batch_entry["target_text"] = target_text
            batch_entry["candidates"] = candidates

        return batch_entry


def get_loader(
    task_list,
    sample_numbers,
    split="toys",
    mode="test",
    batch_size=1,
    workers=4,
    distributed=False,
):

    if split == "yelp":
        from src.data_loader.all_yelp_templates import all_tasks as task_templates

        dataset = P5_Yelp_Dataset(
            task_templates,
            task_list,
            sample_numbers,
            mode=mode,
            split=split,
            rating_augment=False,
        )
    else:
        from src.data_loader.all_amazon_templates import (
            all_tasks as task_templates,
        )

        dataset = P5_Amazon_Dataset(
            task_templates,
            task_list,
            sample_numbers,
            mode=mode,
            split=split,
            rating_augment=False,
        )

    if distributed:
        sampler = DistributedSampler(dataset)
    else:
        sampler = None

    if mode == "train":
        loader = DataLoader(
            dataset,
            batch_size=batch_size,
            shuffle=(sampler is None),
            num_workers=workers,
            pin_memory=True,
            sampler=sampler,
            collate_fn=dataset.collate_fn,
        )
    else:
        loader = DataLoader(
            dataset,
            batch_size=batch_size,
            num_workers=workers,
            pin_memory=True,
            sampler=sampler,
            shuffle=None if (sampler is not None) else False,
            collate_fn=dataset.collate_fn,
            drop_last=False,
        )

    return loader
