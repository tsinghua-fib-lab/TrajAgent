from data_augmentation.da_operator import *

operator_dict = {
    "Ti-insert": Inserter,
    "insert": Inserter,
    "Ti-delete": NotImplementedError,
    "delete": Deleter,
    "Ti-crop": Croper,
    "crop": Croper,
    "Ti-mask": Masker,
    "mask": Masker,
    "Ti-reorder": Reorderer,
    "reorder": Reorderer,
    "Ti-replace": Replacer,
    "replace": Replacer,
    "subset-split": SubsetSplit,
    "slide-window": SlideWindow,
    "recbole-slide-window": RecBoleSlideWindow,
    "cl4srec-crop": CL4SRecCroper,
    "cl4srec-mask": CL4SRecMask,
    "cl4srec-reorder": CL4SRecReorder,
    "cl4srec-mixed-vanilla": CL4SRecVanillaMixed,
    "cl4srec-mixed-all": CL4SRecAllMixed,
}


# Time-interval-aware operations  
ti_operator_dict = [
    "Ti-insert",
    "Ti-delete",
    "Ti-crop",
    "Ti-mask",
    "Ti-reorder",
    "Ti-replace",
    "slide-window",
    "recbole-slide-window",
    "subset-split",
]

# List of operations that need to use their own position sampling methods.
need_unique_position_sampling = []
need_unique_position_sampling += ti_operator_dict

# Dict of the mapping that specific the position sampling method.
unique_position_sampling_mapping = dict()
for each_ops in ti_operator_dict:
    unique_position_sampling_mapping[each_ops] = ["time"]

# List of unique data augmentation
unique_operator_dict = [
    "subset-split",
    "slide-window",
    "recbole-slide-window",
    "cl4srec-crop",
    "cl4srec-mask",
    "cl4srec-reorder",
    "cl4srec-mixed-vanilla",
    "cl4srec-mixed-all",
]
