from scipy.signal import butter, lfilter

def lowpass_filter(audio, sample_rate, cutoff=1000, order=4):
    nyquist = 0.5 * sample_rate
    normal_cutoff = cutoff / nyquist
    b, a = butter(order, normal_cutoff, btype='low', analog=False)
    return lfilter(b, a, audio, axis=0)

def highpass_filter(audio, sample_rate, cutoff=1000, order=4):
    nyquist = 0.5 * sample_rate
    normal_cutoff = cutoff / nyquist
    b, a = butter(order, normal_cutoff, btype='high', analog=False)
    return lfilter(b, a, audio, axis=0)

def bandpass_filter(audio, sample_rate, lowcut=500, highcut=2000, order=4):
    nyquist = 0.5 * sample_rate
    low = lowcut / nyquist
    high = highcut / nyquist
    b, a = butter(order, [low, high], btype='band')
    return lfilter(b, a, audio, axis=0)