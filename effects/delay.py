import numpy as np

def apply_delay(audio, sample_rate, delay_sec=0.3, alpha=0.5):
    delay_samples = int(delay_sec * sample_rate)

    if audio.ndim == 1:
        output = np.zeros(len(audio) + delay_samples)
        output[:len(audio)] += audio
        output[delay_samples:delay_samples + len(audio)] += alpha * audio
    else:
        output = np.zeros((len(audio) + delay_samples, audio.shape[1]))
        output[:len(audio), :] += audio
        output[delay_samples:delay_samples + len(audio), :] += alpha * audio

    output = output / (1 + alpha)

    max_val = np.max(np.abs(output))
    if max_val > 0:
        output = output / max_val

    return output