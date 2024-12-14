from ...m_dependencies import *

def call(self, operation: dict) -> None:
    self.project_data[operation["uid"]] = keras.layers.IntegerLookup(
        max_tokens=(
            operation["args"]["max_tokens"]
            if "max_tokens" in operation["args"]
            else None
        ),
        num_oov_indices=(
            operation["args"]["num_oov_indices"]
            if "num_oov_indices" in operation["args"]
            else 1
        ),
        mask_token=(
            operation["args"]["mask_token"]
            if "mask_token" in operation["args"]
            else None
        ),
        oov_token=(
            operation["args"]["oov_token"] if "oov_token" in operation["args"] else -1
        ),
        vocabulary=(
            self.project_data[operation["args"]["vocabulary"]]
            if "vocabulary" in operation["args"]
            else None
        ),
        vocabulary_dtype=(
            operation["args"]["vocabulary_dtype"]
            if "vocabulary_dtype" in operation["args"]
            else "int64"
        ),
        idf_weights=(
            self.project_data[operation["args"]["idf_weights"]]
            if "idf_weights" in operation["args"]
            else None
        ),
        invert=(
            operation["args"]["invert"] if "invert" in operation["args"] else False
        ),
        output_mode=(
            operation["args"]["output_mode"]
            if "output_mode" in operation["args"]
            else "int"
        ),
        sparse=(
            operation["args"]["sparse"] if "sparse" in operation["args"] else False
        ),
        pad_to_max_tokens=(
            operation["args"]["pad_to_max_tokens"]
            if "pad_to_max_tokens" in operation["args"]
            else False
        ),
        name=(operation["args"]["name"] if "name" in operation["args"] else None),
    )
