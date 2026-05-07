import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker


def _to_mono(audio: np.ndarray) -> np.ndarray:
    """Return a 1-D mono array regardless of input shape."""
    if audio.ndim == 1:
        return audio
    return audio[:, 0] # first channel only


def _magnitude_spectrum(audio: np.ndarray, sr: int):
    """
    Compute a smoothed magnitude spectrum in dB.
    Returns (frequencies_Hz, magnitude_dB).
    """
    mono = _to_mono(audio)
    n = len(mono)

    # Zero-pad to next power of 2 for a faster FFT
    n_fft = 1
    while n_fft < n:
        n_fft <<= 1

    window = np.hanning(n)
    windowed = mono * window

    spectrum = np.fft.rfft(windowed, n=n_fft)
    magnitude = np.abs(spectrum)

    # Normalise so the peak is 0 dB
    magnitude = magnitude / (np.max(magnitude) + 1e-12)
    magnitude_db = 20 * np.log10(magnitude + 1e-12)

    freqs = np.fft.rfftfreq(n_fft, d=1.0 / sr)
    return freqs, magnitude_db


def show_spectrum(audio_before: np.ndarray,
                  audio_after: np.ndarray,
                  sample_rate: int,
                  title: str = "Frequency Spectrum") -> None:
    """
    Open a matplotlib window showing the before/after magnitude spectrum.
    """
    freqs_b, mag_b = _magnitude_spectrum(audio_before, sample_rate)
    freqs_a, mag_a = _magnitude_spectrum(audio_after,  sample_rate)

    fig, ax = plt.subplots(figsize=(10, 4))
    fig.patch.set_facecolor("#1a1208")
    ax.set_facecolor("#231a0e")

    ax.plot(freqs_b, mag_b, color="#7a9aae", linewidth=1.0,
            alpha=0.85, label="Before")
    ax.plot(freqs_a, mag_a, color="#c8a84b", linewidth=1.2,
            alpha=0.95, label="After")

    ax.set_xscale("log")
    ax.set_xlim(20, sample_rate / 2)
    ax.set_ylim(-80, 5)

    ax.set_xlabel("Frequency (Hz)", color="#d0c0a0", fontsize=10)
    ax.set_ylabel("Magnitude (dB)", color="#d0c0a0", fontsize=10)
    ax.set_title(title, color="#c8a84b", fontsize=12, fontweight="bold")

    # Tick formatting
    ax.xaxis.set_major_formatter(
        ticker.FuncFormatter(lambda x, _: f"{int(x):,}" if x >= 1000 else str(int(x)))
    )
    ax.tick_params(colors="#7a6a50")
    for spine in ax.spines.values():
        spine.set_edgecolor("#3a2a10")

    ax.grid(True, which="both", color="#2e2010", linewidth=0.6, linestyle="--")
    ax.legend(facecolor="#1e1810", edgecolor="#3a2a10",
              labelcolor="#d0c0a0", fontsize=9)

    fig.tight_layout()
    plt.show()
