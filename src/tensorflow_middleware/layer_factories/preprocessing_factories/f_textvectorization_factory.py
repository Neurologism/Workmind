from src.tensorflow_middleware.m_dependencies import *


def call(self, operation: dict) -> None:
    self.project_data[operation["uid"]] = keras.layers.TextVectorization(
        max_tokens=(
            operation["args"]["max_tokens"]
            if "max_tokens" in operation["args"]
            else None
        ),
        standardize=(
            operation["args"]["standardize"]
            if "standardize" in operation["args"]
            else "lower_and_strip_punctuation"
        ),
        split=(
            operation["args"]["split"] if "split" in operation["args"] else "whitespace"
        ),
        ngrams=(operation["args"]["ngrams"] if "ngrams" in operation["args"] else None),
        output_mode=(
            operation["args"]["output_mode"]
            if "output_mode" in operation["args"]
            else "int"
        ),
        output_sequence_length=(
            operation["args"]["output_sequence_length"]
            if "output_sequence_length" in operation["args"]
            else None
        ),
        pad_to_max_tokens=(
            operation["args"]["pad_to_max_tokens"]
            if "pad_to_max_tokens" in operation["args"]
            else False
        ),
        vocabulary=(
            self.project_data[operation["args"]["vocabulary"]]
            if "vocabulary" in operation["args"]
            else None
        ),
        idf_weights=(
            self.project_data[operation["args"]["idf_weights"]]
            if "idf_weights" in operation["args"]
            else None
        ),
        sparse=(
            operation["args"]["sparse"] if "sparse" in operation["args"] else False
        ),
        ragged=(
            operation["args"]["ragged"] if "ragged" in operation["args"] else False
        ),
        encoding=(
            operation["args"]["encoding"]
            if "encoding" in operation["args"]
            else "utf-8"
        ),
        name=(operation["args"]["name"] if "name" in operation["args"] else None),
    )
