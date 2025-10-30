import copy
import math
import random


class CL4SRecCroper:
    """
    Refer: CL4SRec.item_crop()
    """

    def __init__(self, items):
        self.items = items
        self._pos_sampler = None

    def init(self, instances, timestamps, **kwargs):
        """
        instances:              (Iterable) the interaction sequence.
        timestamps:             (Iterable) the interaction timestamp sequence.
        """
        pass

    def forward(self, seq, ts, **kwargs):
        """
        seq:                    (Iterable) the interaction sequence.
        ts:                     (Iterable) the interaction timestamp sequence.
        =========
        **kwargs:
        eta:                    (int) the size of cropping.


        pop_counter:            (np.array) the popularity counter. [popularity-position-sampling]

        crop_time_sort:         (str, choice in ["maximum", "minimum"]) [Ti-crop-position-sampling]
        """
        assert "eta" in kwargs, "need to specify eta"
        eta = kwargs["eta"]  # 0.6 by default

        if len(seq) <= 2:
            return [seq], None

        ret_list = []
        for _ in range(2):
            item_list = seq[:-1]  # exclude the target item
            target_item = [seq[-1]]  # target item

            length = len(item_list)
            num_left = math.floor(length * eta)
            crop_begin = random.randint(0, length - num_left)

            if crop_begin + num_left < length:
                croped_item_seq = item_list[crop_begin : crop_begin + num_left]
            else:
                croped_item_seq = item_list[crop_begin:]

            croped_item_seq += target_item
            ret_list.append(croped_item_seq)

        return ret_list, None


class CL4SRecMask:
    """
    Refer: CL4SRec.item_mask()
    """

    def __init__(self, items):
        self.items = items
        self._pos_sampler = None

    def init(self, instances, timestamps, **kwargs):
        """
        instances:              (Iterable) the interaction sequence.
        timestamps:             (Iterable) the interaction timestamp sequence.
        """
        pass

    def forward(self, seq, ts, **kwargs):
        """
        seq:                    (Iterable) the interaction sequence.
        ts:                     (Iterable) the interaction timestamp sequence.
        =========
        **kwargs:
        gamma:                  (int) the size of masking.


        pop_counter:            (np.array) the popularity counter. [popularity-position-sampling]

        crop_time_sort:         (str, choice in ["maximum", "minimum"]) [Ti-crop-position-sampling]
        """
        assert "gamma" in kwargs, "need to specify gamma"
        mask_value = kwargs["cl4srec_mask_value"]
        gamma = kwargs["gamma"]  # 0.3 by default
        elem_type = type(seq[0])
        cvt2int = isinstance(elem_type, type(int))

        if len(seq) <= 2:
            return [seq], None

        ret_list = []
        for _ in range(2):
            item_list = seq[:-1]  # exclude the target item
            target_item = [seq[-1]]  # target item

            length = len(item_list)
            num_mask = math.floor(length * gamma)
            mask_index = random.sample(range(length), k=num_mask)
            masked_item_seq = item_list[:]
            for idx in mask_index:
                masked_item_seq[idx] = str(mask_value) if not cvt2int else mask_value

            masked_item_seq += target_item
            ret_list.append(masked_item_seq)

        return ret_list, None


class CL4SRecReorder:
    """
    Refer: CL4SRec.item_reorder()
    """

    def __init__(self, items):
        self.items = items
        self._pos_sampler = None

    def init(self, instances, timestamps, **kwargs):
        """
        instances:              (Iterable) the interaction sequence.
        timestamps:             (Iterable) the interaction timestamp sequence.
        """
        pass

    def forward(self, seq, ts, **kwargs):
        """
        seq:                    (Iterable) the interaction sequence.
        ts:                     (Iterable) the interaction timestamp sequence.
        =========
        **kwargs:
        beta:                  (int) the size of reorder.


        pop_counter:            (np.array) the popularity counter. [popularity-position-sampling]

        crop_time_sort:         (str, choice in ["maximum", "minimum"]) [Ti-crop-position-sampling]
        """
        assert "beta" in kwargs, "need to specify beta"
        beta = kwargs["beta"]  # 0.6 by default

        if len(seq) <= 2:
            return [seq], None

        item_list = seq[:-1]  # exclude the target item
        target_item = [seq[-1]]  # target item

        ret_list = []
        for i in range(2):
            length = len(item_list)
            num_reorder = math.floor(length * beta)
            reorder_begin = random.randint(0, length - num_reorder)
            shuffle_seq = copy.deepcopy(
                item_list[reorder_begin : reorder_begin + num_reorder]
            )
            random.shuffle(shuffle_seq)
            reordered_item_seq = (
                item_list[:reorder_begin]
                + shuffle_seq
                + item_list[reorder_begin + num_reorder :]
            )

            reordered_item_seq += target_item
            ret_list.append(reordered_item_seq)

        return ret_list, None


class CL4SRecVanillaMixed:
    """
    Refer: CL4SRec.cl4srec_aug()
    """

    def __init__(self, items):
        self.items = items
        self._pos_sampler = None
        self.crop_op = CL4SRecCroper(self.items)
        self.mask_op = CL4SRecMask(self.items)
        self.reorder_op = CL4SRecReorder(self.items)

    def init(self, instances, timestamps, **kwargs):
        """
        instances:              (Iterable) the interaction sequence.
        timestamps:             (Iterable) the interaction timestamp sequence.
        """
        pass

    def forward(self, seq, ts, **kwargs):
        """
        seq:                    (Iterable) the interaction sequence.
        ts:                     (Iterable) the interaction timestamp sequence.
        =========
        **kwargs:
        beta:                  (int) the size of reorder.


        pop_counter:            (np.array) the popularity counter. [popularity-position-sampling]

        crop_time_sort:         (str, choice in ["maximum", "minimum"]) [Ti-crop-position-sampling]
        """
        switch = random.randint(0, 2)

        # Forwarding
        if switch == 0:
            return self.crop_op.forward(seq, ts, **kwargs)
        elif switch == 1:
            return self.mask_op.forward(seq, ts, **kwargs)
        elif switch == 2:
            return self.reorder_op.forward(seq, ts, **kwargs)

        raise RuntimeError("Unreachable")


class CL4SRecAllMixed:
    """
    Refer: CL4SRec.cl4srec_aug()
    """

    def __init__(self, items):
        from data_augmentation.da_operator import (
            Inserter,
            Replacer,
            SubsetSplit,
            Deleter,
        )

        self.items = items
        self.crop_op = CL4SRecCroper(self.items)
        self.mask_op = CL4SRecMask(self.items)
        self.reorder_op = CL4SRecReorder(self.items)
        self.subset_split_op = SubsetSplit(self.items)
        self.insert_op = Inserter(self.items)
        self.replace_op = Replacer(self.items)
        self.delete_op = Deleter(self.items)

        self.aug_list = [
            self.crop_op,
            self.mask_op,
            self.reorder_op,
            self.subset_split_op,
            self.insert_op,
            self.replace_op,
            self.delete_op,
        ]

        self.udf_weights_flag = False
        self.udf_weights = None

    def init(self, instances, timestamps, **kwargs):
        """
        instances:              (Iterable) the interaction sequence.
        timestamps:             (Iterable) the interaction timestamp sequence.
        """
        kwargs["pos"] = "uniform"
        kwargs["select"] = "random"
        # init(self, args, instances, timestamps, uid, traj, **kwargs)
        self.insert_op.init(instances, timestamps, **kwargs)# init(self, args, instances, timestamps, uid, traj, **kwargs)
        self.replace_op.init(instances, timestamps, **kwargs)
        self.delete_op.init(instances, timestamps, **kwargs)

        self.udf_weights_flag = kwargs["cl4srec_mixed_aug_flag"]
        if self.udf_weights_flag:
            dataset_name = kwargs["dataset"]
            weights = kwargs["cl4srec_mixed_aug_weight"][dataset_name]
            weight_keys = [
                "cl4srec_crop",
                "cl4srec_mask",
                "cl4srec_reorder",
                "cl4srec_subset_split",
                "cl4srec_insert",
                "cl4srec_replace",
                "cl4srec_delete",
            ]  # do not change the order
            self.udf_weights = [weights[key] for key in weight_keys]

    def forward(self, seq, ts, **kwargs):
        """
        seq:                    (Iterable) the interaction sequence.
        ts:                     (Iterable) the interaction timestamp sequence.
        =========
        **kwargs:
        beta:                  (int) the size of reorder.


        pop_counter:            (np.array) the popularity counter. [popularity-position-sampling]

        crop_time_sort:         (str, choice in ["maximum", "minimum"]) [Ti-crop-position-sampling]
        """
        if self.udf_weights_flag:
            switch = random.choices(
                range(len(self.aug_list)), weights=self.udf_weights, k=1
            )[0]
        else:
            switch = random.randint(0, len(self.aug_list) - 1)

        if switch == 4:
            kwargs["operation_type"] = "insert"
        elif switch == 5:
            kwargs["operation_type"] = "replace"
        elif switch == 6:
            kwargs["operation_type"] = "delete"
        else:
            pass

        # Forwarding
        return self.aug_list[switch].forward(seq, ts, **kwargs)
