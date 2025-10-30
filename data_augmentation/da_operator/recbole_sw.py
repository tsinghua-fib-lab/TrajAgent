import copy

import numpy as np


class RecBoleSlideWindow:
    def __init__(self, items):
        self.items = items

    def init(self, instances, timestamps, **kwargs):
        return

    def forward(self, seq, ts, **kwargs):
        """
        seq:                    (Iterable) the interaction sequence.
        ts:                     (Iterable) the interaction timestamp sequence.
        =========
        **kwargs:
        start_pos:              (int) the start position for subset split.
        end_pos:                (int) the end position for subset split.
        recbole_sw_length:      (int) the length of the windows
        """
        start_pos = kwargs["start_pos"]
        end_pos = kwargs["end_pos"]
        window_length = kwargs["recbole_sw_length"]

        end_pos = len(seq)

        aug_seqs = []
        aug_ts = None if ts is None else []
        seq_start = 0
        for idx in range(start_pos + 1, end_pos):
            if idx + 1 - seq_start > window_length:
                seq_start += 1

            aug_seqs.append(seq[seq_start:idx] + [seq[idx]])

            if aug_ts is not None:
                aug_ts.append(ts[seq_start:idx] + [ts[idx]])

        return aug_seqs, aug_ts
