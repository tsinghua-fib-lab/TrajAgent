import copy

import numpy as np


class SlideWindow:
    def __init__(self, items):
        self.items = items

    def init(self, args, instances, timestamps, uid, traj, **kwargs):
        return

    def forward(self, args, seq, ts, **kwargs):
        """
        seq:                    (Iterable) the interaction sequence.
        ts:                     (Iterable) the interaction timestamp sequence.
        =========
        **kwargs:
        start_pos:              (int) the start position for subset split.
        end_pos:                (int) the end position for subset split.
        window_length:          (int) the length of the windows
        """
        start_pos = kwargs["start_pos"]
        end_pos = kwargs["end_pos"]
        window_length = kwargs["window_length"]

        if end_pos is None:
            end_pos = len(seq)
        else:
            end_pos = len(seq) + end_pos

        if window_length > end_pos + 1 - start_pos:
            return [seq], [ts]
        elif window_length == end_pos + 1 - start_pos:
            return [copy.deepcopy(seq)], [copy.deepcopy(ts)]

        aug_seqs = []
        aug_ts = None if ts is None else []
        for idx in range(end_pos + 1 - start_pos - window_length):
            aug_seqs.append(seq[idx : idx + window_length])

            if aug_ts is not None:
                aug_ts.append(ts[idx : idx + window_length])

        return aug_seqs, aug_ts, None
