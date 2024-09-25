import numpy as np
import soundfile as sf
from scipy.signal import convolve


# Load audio signal and impulse response
audio_signal, sample_rate = sf.read("data/jazz-piano-24-bits.wav")
impulse_response, _ = sf.read("data/hall-impulse-response.wav")


# Perform convolution
audio_signal = audio_signal / np.abs(audio_signal).max()
impulse_response = impulse_response / np.abs(impulse_response).max() / 0.1
reverb_signal = convolve(audio_signal, impulse_response, mode="full")

# Normalize to prevent clipping
reverb_signal = reverb_signal / np.max(np.abs(reverb_signal))

# Save the reverb audio signal
sf.write("data/jazz-piano-with-reverb.wav", reverb_signal, sample_rate)
