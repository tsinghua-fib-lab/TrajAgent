import importlib
import copy


class Inserter:
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
        pos = "time"
        item_sample_method = args.item_sample
        module_path = "data_augmentation.utils.distribution_sampler"
        if importlib.util.find_spec(module_path):
            module = importlib.import_module(module_path)
            self._pos_sampler = getattr(module, pos + "DistributionPositionSampler")()
            self._item_sampler = getattr(module, item_sample_method + "ItemSampler")(self.items)
            self._item_sampler.init(instances, uid, traj,args=args, **kwargs)
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
        start_pos:              (int) the start position for insertion.
        end_pos:                (int) the end position for insertion.
        insert_nums:            (int) the number of inserted items.
        insert_ratio:           (float) the ratio of inserted items.
        insert_n_times:         (int) the number of insertion for each sequence
        percent_no_augment:     (float) the length of no augmentation at the end of the sequence.

        pop_counter:            (np.array) the popularity counter. [popularity-position-sampling]

        target_item:            (int) the index of the target item. [similar-item-sampling]
        item_embeddings:        (np.array) the item embeddings. [similar-item-sampling]
        op:                     (Callable) the similarity measurement function. [similar-item-sampling]

        insert_time_sort:       (str, choice in ["maximum", "minimum"]) [Ti-insert-position-sampling]
        """
        start_pos = kwargs["start_pos"]
        end_pos = kwargs["end_pos"]
        op_type = kwargs["operation_type"]
        insert_nums = kwargs["insert_nums"]
        insert_ratio = kwargs["insert_ratio"]
        percent_no_augment = kwargs["percent_no_augment"]

        if end_pos is None:
            end_pos = len(seq)
        else:
            end_pos = len(seq) + end_pos
        if percent_no_augment > 0.0:
            no_augment_end_pos = int(len(seq) * (1 - percent_no_augment))
            end_pos = min(no_augment_end_pos, end_pos)

        insert_nums = max(insert_nums, int((end_pos + 1 - start_pos) * insert_ratio))
        if len(seq) < insert_nums + 1:
            return [seq], [ts], args.time_sample

        cvt2int = isinstance(seq[start_pos], int)

        if op_type == "insert":
            aug_seqs = []
            for _ in range(n_times):
                augmented_seq = copy.deepcopy(seq)
                # Note: the 'insert_pos' is an offset to the 'start_pos'
                insert_pos = list(
                    map(
                        lambda x: x + start_pos,
                        self._pos_sampler.sample(
                            seq[start_pos : end_pos + 1], n=insert_nums, **kwargs
                        ),
                    )
                )
                kwargs["insert_pos"] = insert_pos
                select_item = self._item_sampler.sample(
                    seq=seq, n_times=insert_nums, **kwargs
                )
                for pos, each_item in sorted(
                    list(zip(insert_pos, select_item)), reverse=True
                ):
                    augmented_seq.insert(pos, int(each_item) if cvt2int else each_item)
                aug_seqs.append(augmented_seq)
            return aug_seqs, None
        elif op_type == "Ti-insert":
            # For time-interval-aware insert, it's necessary to pass the whole slice ([start_pos: end_pos + 1]).
            aug_seqs = []
            aug_ts = []
            n_times = kwargs["ti_insert_n_times"]
            for _ in range(n_times):
                augmented_seq = copy.deepcopy(seq)
                augmented_ts = copy.deepcopy(ts)
                insert_pos = list(
                    map(
                        lambda x: x + start_pos,
                        self._pos_sampler.sample(
                            args, ts[start_pos : end_pos + 1], n=insert_nums, **kwargs
                        ),
                    )
                )
                kwargs["insert_pos"] = insert_pos
                insert_nums = len(insert_pos)
                select_item = self._item_sampler.sample(
                    seq=seq, n_times=insert_nums, **kwargs
                )
                for pos, each_item in sorted(
                    list(zip(insert_pos, select_item)), reverse=True
                ):
                    augmented_seq.insert(
                        pos + 1, int(each_item) if cvt2int else each_item
                    )
                    augmented_ts.insert(pos + 1, (ts[pos] + ts[pos + 1]) // 2)
                aug_seqs.append(augmented_seq)
                aug_ts.append(augmented_ts)
            return aug_seqs, aug_ts, args.time_sample
        else:
            raise ValueError(f"Invalid operation [{op_type}]")
