from scipy.signal import fftconvolve
import numpy as np

def apply_reverb(audio, impulse_response):
    if audio.ndim == 1:
        output = fftconvolve(audio, impulse_response, mode='full')
    else:
        channels = []
        for ch in range(audio.shape[1]):
            ch_out = fftconvolve(audio[:, ch], impulse_response, mode='full')
            channels.append(ch_out)
        output = np.stack(channels, axis=1)

    max_val = np.max(np.abs(output))
    if max_val > 1:
        output = output / max_val

    return output

def generate_impulse_response(sample_rate, duration=0.3, decay=5.0):
    n = int(duration * sample_rate)
    t = np.linspace(0, duration, n)
    ir = np.exp(-decay * t)
    return ir / np.max(np.abs(ir))