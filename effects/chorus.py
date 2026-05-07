import numpy as np

def apply_chorus(audio, sample_rate, depth_ms=8.0, rate_hz=1.5, base_delay_ms=15.0, mix=0.5):
    if audio.ndim == 1:
        audio = audio[:, np.newaxis]
        mono_input = True
    else:
        mono_input = False

    n_samples, n_channels = audio.shape

    base_delay_samples = base_delay_ms * sample_rate / 1000.0
    depth_samples = depth_ms * sample_rate / 1000.0

    max_delay = int(np.ceil(base_delay_samples + depth_samples)) + 2
    padded = np.pad(audio, ((max_delay, 0), (0, 0)), mode='constant')

    output = np.zeros_like(audio)

    t = np.arange(n_samples) / sample_rate
    lfo = np.sin(2 * np.pi * rate_hz * t)

    for n in range(n_samples):
        current_delay = base_delay_samples + depth_samples * lfo[n]

        read_index = n + max_delay - current_delay
        i0 = int(np.floor(read_index))
        i1 = i0 + 1
        frac = read_index - i0

        delayed_sample = (1 - frac) * padded[i0, :] + frac * padded[i1, :]
        output[n, :] = (1 - mix) * audio[n, :] + mix * delayed_sample

    max_val = np.max(np.abs(output))
    if max_val > 0:
        output = output / max_val

    if mono_input:
        return output[:, 0]
    return output