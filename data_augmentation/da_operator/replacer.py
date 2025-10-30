import importlib
import copy


class Replacer:
    def __init__(self, items):
        self.items = items
        self._pos_sampler = None
        self._item_sampler = None

    def init(self, args, instances, timestamps, uid, traj, **kwargs):
        """
        instances:              (Iterable) the interaction sequence.
        timestamps:             (Iterable) the interaction timestamp sequence.

        pos:                    (str, choice in ['uniform']) position sampling method.
        select:                 (str, choice in ['random', 'memorybased']) item sampling method.
        """
        pos = kwargs["pos"]
        item_sample_method = args.item_sample
        module_path = "data_augmentation.utils.distribution_sampler"

        if importlib.util.find_spec(module_path):
            module = importlib.import_module(module_path)
            self._pos_sampler = getattr(module, pos + "DistributionPositionSampler")()
            self._item_sampler = getattr(module, item_sample_method + "ItemSampler")(self.items)
            self._item_sampler.init(instances, uid, traj, args=args, **kwargs)
        else:
            raise ValueError(
                f"Invalid argument 'pos'[{pos}], it must be one of 'uniform', 'popularity', 'distance'."
            )

        if self._pos_sampler is None:
            raise ValueError(
                f"Invalid argument 'pos'[{pos}], it must be one of 'uniform', 'popularity', 'distance'."
            )

        if self._item_sampler is None:
            raise ValueError(
                f"Invalid argument 'select'[{pos}], it must be one of 'random', 'similar', 'unvisited', 'redundant'."
            )

    def forward(self, args, seq, ts, **kwargs):
        """
        **kwargs:
        start_pos:              (int) the start position for replacing.
        end_pos:                (int) the end position for replacing.
        replace_nums:           (int) the number of replaced items.
        replace_ratio:          (float) the ratio of replaced items.
        replace_n_times:        (int) the number of replacement for each sequence

        pop_counter:            (np.array) the popularity counter. [popularity-position-sampling]

        target_item:            (int) the index of the target item. [similar-item-sampling]
        item_embeddings:        (np.array) the item embeddings. [similar-item-sampling]
        op:                     (Callable) the similarity measurement function. [similar-item-sampling]
        """
        # 设置默认值，避免 KeyError
        kwargs.setdefault("replace_time_sort", "minimum")
        kwargs.setdefault("ti_replace_n_times", 1)
        kwargs.setdefault("replace_nums", 1)
        start_pos = kwargs["start_pos"]
        end_pos = kwargs["end_pos"]
        replace_nums = kwargs["replace_nums"]
        replace_ratio = kwargs["replace_ratio"]

        if end_pos is None:
            end_pos = len(seq)
        else:
            end_pos = len(seq) + end_pos

        replace_nums = max(replace_nums, int((end_pos + 1 - start_pos) * replace_ratio))
        if len(seq) < replace_nums + 1:
            return [seq], [ts], kwargs.get("replace_time_sort", "minimum")

        op_type = kwargs["operation_type"]

        cvt2int = isinstance(seq[start_pos], int)

        if op_type == "replace":
            aug_seqs = []
            for _ in range(n_times):
                # Note: the 'replace_pos' is an offset to the 'start_pos'
                replace_pos = list(
                    map(
                        lambda x: x + start_pos,
                        self._pos_sampler.sample(
                            args=args, seq=seq[start_pos:end_pos], n=replace_nums, **kwargs
                        ),
                    )
                )
                kwargs["insert_pos"] = replace_pos
                select_item = self._item_sampler.sample(
                    seq=seq, n_times=replace_nums, **kwargs
                )
                replace_seq = copy.deepcopy(seq)
                for pos, each_item in zip(replace_pos, select_item):
                    replace_seq[pos] = each_item if cvt2int else each_item
                aug_seqs.append(replace_seq)
            return aug_seqs, None
        elif op_type == "Ti-replace":
            aug_seqs = []
            aug_ts = []
            n_times = kwargs["ti_replace_n_times"]
            # 设置默认值，避免 KeyError
            replace_time_sort = kwargs.get("replace_time_sort", "minimum")      
            for _ in range(n_times):
                # Adopt the Ti-mask position sampler for convenience.
                pos_sampler_args = {
                    "operation_type": "Ti-mask",
                    "pos": "time",
                    "start_pos": kwargs["start_pos"],
                    "end_pos": kwargs["end_pos"],
                    "mask_nums": replace_nums,
                    "mask_ratio": 0,
                    "mask_time_sort": replace_time_sort,
                }
                replace_pos = list(
                    map(
                        lambda x: x + start_pos,
                        self._pos_sampler.sample(
                           args=args,seq=ts[start_pos:end_pos], n=replace_nums, **pos_sampler_args
                        ),
                    )
                )
                # 确保位置数量匹配
                if len(replace_pos) != replace_nums:
                    # 如果位置数量不足，调整 replace_nums
                    replace_nums = len(replace_pos)
                    if replace_nums == 0:
                        # 如果没有可用位置，跳过这个增强
                        aug_seqs.append(copy.deepcopy(seq))
                        aug_ts.append(copy.deepcopy(ts))
                        continue
                kwargs["insert_pos"] = replace_pos
                select_item = self._item_sampler.sample(
                    seq=seq, n_times=replace_nums, **kwargs
                )
                replace_seq = copy.deepcopy(seq)
                for pos, each_item in zip(replace_pos, select_item):
                    replace_seq[pos] = int(each_item) if cvt2int else each_item
                aug_seqs.append(replace_seq)
                aug_ts.append(copy.deepcopy(ts))
            return aug_seqs, aug_ts, pos_sampler_args["mask_time_sort"]
        else:
            raise ValueError(f"Invalid operation [{op_type}]")
