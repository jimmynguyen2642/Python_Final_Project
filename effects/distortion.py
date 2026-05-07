import numpy as np

def apply_distortion(audio, gain=2.0, mix=1.0):
    distorted = np.tanh(gain * audio)
    output = (1 - mix) * audio + mix * distorted

    max_val = np.max(np.abs(output))
    if max_val > 0:
        output = output / max_val

    return output