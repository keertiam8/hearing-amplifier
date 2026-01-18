import numpy as np
import sounddevice as sd
import matplotlib.pyplot as plt
import json
import importlib
from scipy.signal import butter, lfilter

sd.default.device = (15, 12)
print("🎧 Using input device:", sd.query_devices(15)['name'])
print("🔊 Using output device:", sd.query_devices(12)['name'])

try:
    from vosk import Model, KaldiRecognizer
    has_vosk = True
except Exception:
    has_vosk = False
    print("⚠️ vosk package not installed — local recognition disabled. Install with: pip install vosk")

try:
    librosa = importlib.import_module("librosa")
except Exception:
    librosa = None
    print("⚠️ Librosa not found — install it with: pip install librosa")

mic_rate = 16000
vosk_rate = 16000
duration = 0.5
device = 15
N = int(mic_rate * duration)

def butter_bandpass(lowcut, highcut, fs, order=5):
    nyq = 0.5 * fs
    low = lowcut / nyq
    high = highcut / nyq
    b, a = butter(order, [low, high], btype='band')
    return b, a

def apply_bandpass_filter(data, lowcut=100.0, highcut=4000.0, fs=48000, order=5):
    b, a = butter_bandpass(lowcut, highcut, fs, order=order)
    return lfilter(b, a, data)

enable_hearing_aid = True
hearing_low = 300
hearing_high = 3000
gain = 3.0

def hearing_aid_process(data, fs):
    filtered = apply_bandpass_filter(data, lowcut=hearing_low, highcut=hearing_high, fs=fs)
    amplified = filtered * gain
    return np.clip(amplified, -1.0, 1.0)

recognizer = None
if has_vosk:
    try:
        model = Model("vosk-model-small-en-us-0.15")
        recognizer = KaldiRecognizer(model, vosk_rate)
        print("✅ Vosk model loaded successfully.")
    except Exception as e:
        recognizer = None
        print("❌ Failed to load Vosk:", e)

try:
    from google.cloud import speech as gspeech
    google_client = gspeech.SpeechClient()
    use_google = True
    print("✅ Google Cloud Speech available.")
except:
    google_client = None
    use_google = False

plt.ion()
fig, ax = plt.subplots()
x = np.fft.rfftfreq(N, d=1/mic_rate)
line, = ax.plot(x, np.zeros_like(x))
ax.set_xlim(0, 4000)
ax.set_ylim(0, 1)
ax.set_xlabel("Frequency (Hz)")
ax.set_ylabel("Amplitude")
plt.show(block=False)

audio_buffer = np.zeros(N, dtype=np.float32)

def audio_callback(indata, frames, time, status):
    global audio_buffer
    if status:
        print(status)
    audio_buffer = np.roll(audio_buffer, -frames)
    audio_buffer[-frames:] = indata[:, 0]

dev_info = sd.query_devices(device)
try:
    sd.check_input_settings(device=device, channels=1, samplerate=mic_rate)
except Exception as e:
    dev_default = dev_info.get('default_samplerate')
    if dev_default is not None:
        print(f"⚠️ Requested sample rate {mic_rate} Hz not supported by device {device}: {e}")
        mic_rate = int(dev_default)
        print(f"➡️ Falling back to device default samplerate: {mic_rate} Hz")
        N = int(mic_rate * duration)
        audio_buffer = np.zeros(N, dtype=np.float32)
        x = np.fft.rfftfreq(N, d=1/mic_rate)
        line.set_xdata(x)
        line.set_ydata(np.zeros_like(x))
    else:
        raise

with sd.InputStream(samplerate=mic_rate, channels=1, device=device, callback=audio_callback):
    print("🎙️ Speak... press Ctrl+C to stop.")
    try:
        while True:
            filtered_audio = apply_bandpass_filter(audio_buffer, 100, 4000, mic_rate)
            fft_data = np.abs(np.fft.rfft(filtered_audio)) / N
            fft_data *= 10
            line.set_ydata(fft_data)
            ax.set_ylim(0, max(1, float(fft_data.max()) * 1.1))
            plt.pause(0.01)

            if enable_hearing_aid:
                amplified_audio = hearing_aid_process(filtered_audio, mic_rate)
                if librosa:
                    amplified_resampled = librosa.resample(amplified_audio, orig_sr=mic_rate, target_sr=48000)
                    sd.play(amplified_resampled, 48000, blocking=False)
                else:
                    sd.play(amplified_audio, 48000, blocking=False)

    except KeyboardInterrupt:
        print("\n🛑 Stopped.")
        plt.ioff()
        plt.show()
