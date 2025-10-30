import importlib
import copy
import numpy as np


class Deleter:
    def __init__(self, items):
        self.items = items
        self._pos_sampler = None

    def init(self, instances, timestamps, **kwargs):
        """
        instances:              (Iterable) the interaction sequence.
        timestamps:             (Iterable) the interaction timestamp sequence.

        pos:                    (str, choice in ['uniform']) position sampling method.
        """
        pos = kwargs["pos"]

        module_path = "data_augmentation.utils.distribution_sampler"
        if importlib.util.find_spec(module_path):
            module = importlib.import_module(module_path)
            self._pos_sampler = getattr(module, pos + "DistributionPositionSampler")()
        else:
            raise ValueError(
                f"Invalid argument 'pos'[{pos}], it must be one of 'uniform', 'popularity', 'distance'."
            )

        if self._pos_sampler is None:
            raise ValueError(
                f"Invalid argument 'pos'[{pos}], it must be one of 'uniform', 'popularity', 'distance'."
            )

    def forward(self, seq, ts, **kwargs):
        """
        seq:                    (Iterable) the interaction sequence.
        ts:                     (Iterable) the interaction timestamp sequence.
        =========
        **kwargs:
        start_pos:              (int) the start position for deletion.
        end_pos:                (int) the end position for deletion.
        delete_nums:            (int) the number of deleted items.
        delete_ratio:           (float) the ratio of deleted items.
        delete_n_times:         (int) the number of deletion for each sequence
        ti_delete_n_times:      (int)

        pop_counter:            (np.array) the popularity counter. [popularity-position-sampling]
        """
        start_pos = kwargs["start_pos"]
        end_pos = kwargs["end_pos"]
        delete_ratio = kwargs["delete_ratio"]
        delete_nums = kwargs["delete_nums"]
        n_times = kwargs["delete_n_times"]

        if len(seq) == 2:
            return [], []

        if end_pos is None:
            end_pos = len(seq)
        else:
            end_pos = len(seq) + end_pos

        delete_nums = max(delete_nums, int((end_pos + 1 - start_pos) * delete_ratio))

        op_type = kwargs["operation_type"]

        if op_type == "delete":
            aug_seqs = []
            for _ in range(n_times):
                # Note: the 'delete_pos' is an offset to the 'start_pos'
                delete_pos = list(
                    map(
                        lambda x: x + start_pos,
                        self._pos_sampler.sample(
                            seq[start_pos:end_pos], n=delete_nums, **kwargs
                        ),
                    )
                )
                aug_seqs.append(np.delete(copy.deepcopy(seq), delete_pos).tolist())
            return aug_seqs, None
        elif op_type == "Ti-delete":
            aug_seqs = []
            aug_ts = []
            n_times = kwargs["ti_delete_n_times"]
            for _ in range(n_times):
                delete_pos = list(
                    map(
                        lambda x: x + start_pos,
                        self._pos_sampler.sample(
                            ts[start_pos:end_pos], n=delete_nums, **kwargs
                        ),
                    )
                )
                aug_seqs.append(np.delete(copy.deepcopy(seq), delete_pos).tolist())
                aug_ts.append(np.delete(copy.deepcopy(ts), delete_pos).tolist())
            return aug_seqs, aug_ts
        else:
            raise ValueError(f"Invalid operation [{op_type}]")
