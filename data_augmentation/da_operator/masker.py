import importlib
import copy


class Masker:
    def __init__(self, items):
        self.items = items
        self._pos_sampler = None

    def init(self, args, instances, timestamps, uid, traj, **kwargs):
            """
            instances:              (Iterable) the interaction sequence.
            timestamps:             (Iterable) the interaction timestamp sequence.

            pos:                    (str, choice in ['uniform']) position sampling method.
            select:                 (str, choice in ['random', 'memorybased']) item sampling method.
            """
            pos = kwargs["pos"]
            select =args.item_sample

            module_path = "data_augmentation.utils.distribution_sampler"
            if importlib.util.find_spec(module_path):
                module = importlib.import_module(module_path)
                self._pos_sampler = getattr(module, pos + "DistributionPositionSampler")()
                self._item_sampler = getattr(module, select + "ItemSampler")(self.items)
                    # (self, instances, traj, **kwargs):
                # self._item_sampler.init(instances, uid, traj, **kwargs)
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
        seq:                    (Iterable) the interaction sequence.
        ts:                     (Iterable) the interaction timestamp sequence.
        =========
        **kwargs:
        start_pos:              (int) the start position for masking.
        end_pos:                (int) the end position for masking.
        mask_nums:              (int) the number of masked items.
        mask_ratio:             (float) the ratio of masked items.
        mask_value:             (int) the value of masked items.
        mask_n_times:           (int) the number of mask for each sequence

        pop_counter:            (np.array) the popularity counter. [popularity-position-sampling]

        mask_time_sort:         (str, choice in ["maximum", "minimum"]) [Ti-mask-position-sampling]
        """
        kwargs.setdefault("mask_nums", 1)
        kwargs.setdefault("mask_ratio", 0)
        kwargs.setdefault("mask_time_sort", "minimum")
        kwargs.setdefault("mask_value", 1)
        kwargs.setdefault("ti_mask_n_times", 1)
        
        start_pos = kwargs["start_pos"]
        end_pos = kwargs["end_pos"]
        mask_nums = kwargs["mask_nums"]
        mask_ratio = kwargs["mask_ratio"]
        mask_value = kwargs["mask_value"]

        if end_pos is None:
            end_pos = len(seq)
        else:
            end_pos = len(seq) + end_pos

        mask_nums = max(mask_nums, int((end_pos + 1 - start_pos) * mask_ratio))
        
        if len(seq) < mask_nums:
            return [seq], [ts],  args.time_sample

        op_type = kwargs["operation_type"]

        cvt2int = isinstance(seq[start_pos], int)

        if op_type == "mask":
            aug_seqs = []
            for _ in range(n_times):
                # Note: the 'mask_pos' is an offset to the 'start_pos'
                mask_pos = list(
                    map(
                        lambda x: x + start_pos,
                        self._pos_sampler.sample(
                            args=args, seq=seq[start_pos:end_pos], n=mask_nums, **kwargs
                        ),
                    )
                )
                mask_seq = copy.deepcopy(seq)
                for candidate_pos in mask_pos:
                    mask_seq[candidate_pos] = (
                        str(mask_value) if not cvt2int else mask_value
                    )
                aug_seqs.append(mask_seq)
            return aug_seqs, None
        elif op_type == "Ti-mask":
            aug_seqs = []
            aug_ts = []
            n_times = kwargs["ti_mask_n_times"]
            for _ in range(n_times):
                mask_pos = list(
                    map(
                        lambda x: x + start_pos,
                        self._pos_sampler.sample(
                            args=args, seq=ts[start_pos:end_pos], n=mask_nums, **kwargs
                        ),
                    )
                )
                mask_seq = copy.deepcopy(seq)
                for candidate_pos in mask_pos:
                    mask_seq[candidate_pos] = (
                        str(mask_value) if not cvt2int else mask_value
                    )
                aug_seqs.append(mask_seq)
                aug_ts.append(copy.deepcopy(ts))
            return aug_seqs, aug_ts, args.time_sample
        else:
            raise ValueError(f"Invalid operation [{op_type}]")
