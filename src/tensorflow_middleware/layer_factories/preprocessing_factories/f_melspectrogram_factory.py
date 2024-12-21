from ...m_dependencies import *


def call(self, operation: dict) -> None:
    self.project_data[operation["uid"]] = keras.layers.MelSpectrogram(
        fft_length=(
            operation["args"]["fft_length"]
            if "fft_length" in operation["args"]
            else 2048
        ),
        sequence_stride=(
            operation["args"]["sequence_stride"]
            if "sequence_stride" in operation["args"]
            else 512
        ),
        sequence_length=(
            operation["args"]["sequence_length"]
            if "sequence_length" in operation["args"]
            else None
        ),
        window=(
            operation["args"]["window"] if "window" in operation["args"] else "hann"
        ),
        sampling_rate=(
            operation["args"]["sampling_rate"]
            if "sampling_rate" in operation["args"]
            else 16000
        ),
        num_mel_bins=(
            operation["args"]["num_mel_bins"]
            if "num_mel_bins" in operation["args"]
            else 128
        ),
        min_freq=(
            operation["args"]["min_freq"] if "min_freq" in operation["args"] else 20.0
        ),
        max_freq=(
            operation["args"]["max_freq"] if "max_freq" in operation["args"] else None
        ),
        power_to_db=(
            operation["args"]["power_to_db"]
            if "power_to_db" in operation["args"]
            else False
        ),
        top_db=(operation["args"]["top_db"] if "top_db" in operation["args"] else 80.0),
        mag_exp=(
            operation["args"]["mag_exp"] if "mag_exp" in operation["args"] else 2.0
        ),
        min_power=(
            operation["args"]["min_power"]
            if "min_power" in operation["args"]
            else 1e-10
        ),
        ref_power=(
            operation["args"]["ref_power"] if "ref_power" in operation["args"] else 1.0
        ),
    )
