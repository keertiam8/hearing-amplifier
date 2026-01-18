#!/usr/bin/env python3

import tkinter as tk
import tkinter.ttk as ttk
import numpy as np
import sounddevice as sd
from scipy.signal import butter, lfilter
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from threading import Thread
import importlib
import time

import core
from que import Queue, Pair

# Try loading librosa for resampling
try:
    librosa = importlib.import_module("librosa")
except Exception:
    librosa = None
    print("⚠️ Librosa not found — install it with: pip install librosa")


# ------------------ Helper Functions for Audio Processing ------------------
def butter_bandpass(lowcut, highcut, fs, order=5):
    nyq = 0.5 * fs
    low = lowcut / nyq
    high = highcut / nyq
    b, a = butter(order, [low, high], btype='band')
    return b, a


def apply_bandpass_filter(data, lowcut=100.0, highcut=4000.0, fs=48000, order=5):
    b, a = butter_bandpass(lowcut, highcut, fs, order=order)
    return lfilter(b, a, data)


def hearing_aid_process(data, fs, hearing_low=300, hearing_high=3000, gain=3.0):
    filtered = apply_bandpass_filter(data, lowcut=hearing_low, highcut=hearing_high, fs=fs)
    amplified = filtered * gain
    return np.clip(amplified, -1.0, 1.0)


# ------------------ Text Widget for Transcription Display ------------------
class Text(tk.Text):
    def __init__(self, master):
        super().__init__(master)
        self.res_queue = Queue[Pair]()
        self.tag_config("done", foreground="black")
        self.tag_config("curr", foreground="blue", underline=True)
        self.insert("end", "  ", "done")
        self.record = self.index("end-1c")
        self.see("end")
        self.config(state="disabled")
        self.update()

    def update(self):
        while self.res_queue:
            self.config(state="normal")
            if res := self.res_queue.get():
                done = res.done
                curr = res.curr
                self.delete(self.record, "end")
                self.insert("end", done, "done")
                self.record = self.index("end-1c")
                self.insert("end", curr, "curr")
            else:
                done = self.get(self.record, "end-1c")
                self.delete(self.record, "end")
                self.insert("end", done, "done")
                self.insert("end", "\n", "done")
                self.insert("end", "  ", "done")
                self.record = self.index("end-1c")
            self.see("end")
            self.config(state="disabled")
        self.after(100, self.update)  # avoid busy waiting


class IntegratedApp(tk.Tk):
    def __init__(self):
        super().__init__()
        self._proc = None
        self._audio_running = False
        self._audio_thread = None
        self._input_stream = None
        self._output_stream = None
        
        self.mic_rate = 16000
        self.output_rate = 48000  
        self.duration = 0.5
        self.input_device = 15  
        self.output_device = 12  
        self.N = int(self.mic_rate * self.duration)
        self.audio_buffer = np.zeros(self.N, dtype=np.float32)
        self.output_buffer = np.zeros(int(self.output_rate * 0.1), dtype=np.float32)  # 100ms buffer
        self.enable_hearing_aid = tk.BooleanVar(value=True)
        
        self.title("Integrated Audio Processing & Transcription")
        self.geometry("1200x800")
        
        self.top_frame = ttk.Frame(self)
        self.mid_frame = ttk.Frame(self)
        self.bot_frame = ttk.Frame(self)
        self.viz_frame = ttk.Frame(self)
        
        self.top_frame.grid(row=0, column=0, columnspan=2, sticky="ew", padx=5, pady=5)
        self.mid_frame.grid(row=1, column=0, columnspan=2, sticky="ew", padx=5, pady=5)
        self.viz_frame.grid(row=2, column=0, columnspan=2, sticky="nsew", padx=5, pady=5)
        self.bot_frame.grid(row=3, column=0, columnspan=2, sticky="ew", padx=5, pady=5)
        
        self.columnconfigure(0, weight=1)
        self.columnconfigure(1, weight=1)
        self.rowconfigure(2, weight=1)
        
        self.setup_audio_controls()
        self.setup_transcription_controls()
        self.setup_visualization()
        self.setup_transcription_display()
        
        self.start_audio_visualization()
        
    def setup_audio_controls(self):
        """Setup audio processing controls"""
        ttk.Label(self.top_frame, text="Input Device:").pack(side="left", padx=(5, 5))
        
        self.input_device_spin = ttk.Spinbox(self.top_frame, from_=0, to=50, increment=1, width=5, state="readonly")
        self.input_device_spin.set(self.input_device)
        self.input_device_spin.pack(side="left", padx=(0, 10))
        
        ttk.Label(self.top_frame, text="Output Device:").pack(side="left", padx=(5, 5))
        
        self.output_device_spin = ttk.Spinbox(self.top_frame, from_=0, to=50, increment=1, width=5, state="readonly")
        self.output_device_spin.set(self.output_device)
        self.output_device_spin.pack(side="left", padx=(0, 10))
        
        ttk.Checkbutton(self.top_frame, text="Enable Hearing Aid", variable=self.enable_hearing_aid).pack(side="left", padx=(5, 10))
        
        ttk.Label(self.top_frame, text="Sample Rate:").pack(side="left", padx=(5, 5))
        self.rate_label = ttk.Label(self.top_frame, text=f"{self.mic_rate} Hz")
        self.rate_label.pack(side="left", padx=(0, 10))
        
    def setup_transcription_controls(self):
        """Setup transcription and translation controls"""
        ttk.Label(self.mid_frame, text="Mic:").pack(side="left", padx=(5, 5))
        self.mic_combo = ttk.Combobox(self.mid_frame, values=["default"], state="readonly", width=15)
        self.mic_combo.current(0)
        self.mic_combo.pack(side="left", padx=(0, 5))
        
        ttk.Button(self.mid_frame, text="Refresh", 
                  command=lambda: self.mic_combo.config(values=["default"] + core.get_mic_names())).pack(side="left", padx=(0, 10))
        
        ttk.Label(self.mid_frame, text="Model:").pack(side="left", padx=(5, 5))
        self.model_combo = ttk.Combobox(self.mid_frame, values=core.MODELS, state="normal", width=10)
        self.model_combo.set("small")
        self.model_combo.pack(side="left", padx=(0, 5))
        
        ttk.Label(self.mid_frame, text="Device:").pack(side="left", padx=(5, 5))
        self.device_combo = ttk.Combobox(self.mid_frame, values=core.DEVICES, state="readonly", width=8)
        self.device_combo.current(0)
        self.device_combo.pack(side="left", padx=(0, 5))
        
        self.vad_check = ttk.Checkbutton(self.mid_frame, text="VAD", onvalue=True, offvalue=False)
        self.vad_check.state(("!alternate", "selected"))
        self.vad_check.pack(side="left", padx=(0, 10))
        
    def setup_visualization(self):
        """Setup FFT visualization"""
        self.fig, self.ax = plt.subplots(figsize=(10, 3))
        x = np.fft.rfftfreq(self.N, d=1/self.mic_rate)
        self.line, = self.ax.plot(x, np.zeros_like(x))
        self.ax.set_xlim(0, 4000)
        self.ax.set_ylim(0, 1)
        self.ax.set_xlabel("Frequency (Hz)")
        self.ax.set_ylabel("Amplitude")
        self.ax.set_title("Real-time FFT Audio Spectrum")
        self.ax.grid(True, alpha=0.3)
        
        self.canvas = FigureCanvasTkAgg(self.fig, master=self.viz_frame)
        self.canvas.draw()
        self.canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)
        
    def setup_transcription_display(self):
        """Setup transcription and translation text displays"""
        ttk.Label(self.bot_frame, text="Source:").pack(side="left", padx=(5, 5))
        self.source_combo = ttk.Combobox(self.bot_frame, values=["auto"] + core.LANGS, state="readonly", width=8)
        self.source_combo.current(0)
        self.source_combo.pack(side="left", padx=(0, 5))
        
        ttk.Label(self.bot_frame, text="Target:").pack(side="left", padx=(5, 5))
        self.target_combo = ttk.Combobox(self.bot_frame, values=["none"] + core.LANGS, state="readonly", width=8)
        self.target_combo.current(0)
        self.target_combo.pack(side="left", padx=(0, 5))
        
        ttk.Label(self.bot_frame, text="Prompt:").pack(side="left", padx=(5, 5))
        self.prompt_entry = ttk.Entry(self.bot_frame, state="normal", width=20)
        self.prompt_entry.pack(side="left", padx=(0, 5), fill="x", expand=True)
        
        self.control_button = ttk.Button(self.bot_frame)
        self.control_button.pack(side="left", padx=(5, 5))
        self.on_stopped()
        
        text_frame = ttk.Frame(self)
        text_frame.grid(row=4, column=0, columnspan=2, sticky="nsew", padx=5, pady=5)
        self.rowconfigure(4, weight=1)
        
        ttk.Label(text_frame, text="Transcription:").grid(row=0, column=0, sticky="w", padx=5, pady=2)
        self.ts_text = Text(text_frame)
        self.ts_text.grid(row=1, column=0, sticky="nsew", padx=5, pady=5)
        
        ttk.Label(text_frame, text="Translation:").grid(row=0, column=1, sticky="w", padx=5, pady=2)
        self.tl_text = Text(text_frame)
        self.tl_text.grid(row=1, column=1, sticky="nsew", padx=5, pady=5)
        
        text_frame.columnconfigure(0, weight=1)
        text_frame.columnconfigure(1, weight=1)
        text_frame.rowconfigure(1, weight=1)
        
    def audio_callback(self, indata, frames, time, status):
        """Audio input callback"""
        if status:
            print(status)
        self.audio_buffer = np.roll(self.audio_buffer, -frames)
        self.audio_buffer[-frames:] = indata[:, 0]
    
    def output_callback(self, outdata, frames, time, status):
        """Audio output callback - plays the processed audio"""
        if status:
            print(status)
        
        outdata[:] = self.output_buffer[:frames].reshape(-1, 1)
        self.output_buffer = np.roll(self.output_buffer, -frames)
        self.output_buffer[-frames:] = 0  # Fill with silence
        
    def start_audio_visualization(self):
        """Start the audio visualization loop"""
        self._audio_running = True
        self._audio_thread = Thread(target=self.audio_visualization_loop, daemon=True)
        self._audio_thread.start()
        
    def audio_visualization_loop(self):
        """Run audio visualization and processing"""
        try:
            dev_info = sd.query_devices(self.input_device)
            try:
                sd.check_input_settings(device=self.input_device, channels=1, samplerate=self.mic_rate)
            except Exception as e:
                dev_default = dev_info.get('default_samplerate')
                if dev_default is not None:
                    print(f"⚠️ Requested sample rate {self.mic_rate} Hz not supported: {e}")
                    self.mic_rate = int(dev_default)
                    print(f"➡️ Falling back to device default: {self.mic_rate} Hz")
                    self.N = int(self.mic_rate * self.duration)
                    self.audio_buffer = np.zeros(self.N, dtype=np.float32)
                    self.after(0, lambda: self.rate_label.config(text=f"{self.mic_rate} Hz"))
                    x = np.fft.rfftfreq(self.N, d=1/self.mic_rate)
                    self.line.set_xdata(x)
                    self.line.set_ydata(np.zeros_like(x))
                else:
                    raise
            
            sd.default.device = (self.input_device, self.output_device)
            print(f"🎧 Using input device: {sd.query_devices(self.input_device)['name']}")
            print(f"🔊 Using output device: {sd.query_devices(self.output_device)['name']}")
            
            self._input_stream = sd.InputStream(
                samplerate=self.mic_rate, 
                channels=1, 
                device=self.input_device, 
                callback=self.audio_callback
            )
            
            self._output_stream = sd.OutputStream(
                samplerate=self.output_rate,
                channels=1,
                device=self.output_device,
                callback=self.output_callback
            )
            
            self._input_stream.start()
            self._output_stream.start()
            
            print("🎙️ Audio visualization started...")
            
            while self._audio_running:
                try:
                    filtered_audio = apply_bandpass_filter(self.audio_buffer, 100, 4000, self.mic_rate)
                    
                    fft_data = np.abs(np.fft.rfft(filtered_audio)) / self.N
                    fft_data *= 10
                    
                    self.line.set_ydata(fft_data)
                    self.ax.set_ylim(0, max(1, float(fft_data.max()) * 1.1))
                    self.canvas.draw()
                    self.canvas.flush_events()
                    
                    if self.enable_hearing_aid.get():
                        amplified_audio = hearing_aid_process(filtered_audio, self.mic_rate)
                        
                        if np.abs(amplified_audio).max() > 0.01:
                            if librosa:
                                amplified_resampled = librosa.resample(
                                    amplified_audio, 
                                    orig_sr=self.mic_rate, 
                                    target_sr=self.output_rate
                                )
                            else:
                                resample_ratio = self.output_rate / self.mic_rate
                                new_length = int(len(amplified_audio) * resample_ratio)
                                amplified_resampled = np.interp(
                                    np.linspace(0, len(amplified_audio), new_length),
                                    np.arange(len(amplified_audio)),
                                    amplified_audio
                                )
                            
                            buffer_len = min(len(amplified_resampled), len(self.output_buffer))
                            self.output_buffer[:buffer_len] = amplified_resampled[:buffer_len]
                    
                    self.update_idletasks()
                    
                    time.sleep(0.05)
                    
                except Exception as viz_error:
                    print(f"⚠️ Visualization frame error: {viz_error}")
                    
        except Exception as e:
            print(f"❌ Audio visualization error: {e}")
            import traceback
            traceback.print_exc()
        finally:
            if self._input_stream:
                self._input_stream.stop()
                self._input_stream.close()
            if self._output_stream:
                self._output_stream.stop()
                self._output_stream.close()
            print("Audio visualization stopped.")
            
    def on_started(self, proc: core.Processor):
        """Called when transcription starts"""
        self._proc = proc
        
        def stop():
            self.control_button.config(text="Stopping...", state="disabled")
            proc.stop(self.on_stopped)

        self.control_button.config(text="Stop", command=stop, state="normal")

    def on_stopped(self, err: Exception | None = None):
        """Called when transcription stops"""
        if err:
            print(err)
        
        self._proc = None

        def start():
            self.control_button.config(text="Starting...", state="disabled")
            self.input_device = int(self.input_device_spin.get())
            self.output_device = int(self.output_device_spin.get())
            
            core.start(
                index=None if self.mic_combo.current() == 0 else self.mic_combo.current() - 1,
                model=self.model_combo.get(),
                device=self.device_combo.get(),
                vad=self.vad_check.instate(("selected",)),
                prompts=[self.prompt_entry.get()],
                memory=3,
                patience=5.0,
                timeout=5.0,
                source=None if self.source_combo.get() == "auto" else self.source_combo.get(),
                target=None if self.target_combo.get() == "none" else self.target_combo.get(),
                tsres_queue=self.ts_text.res_queue,
                tlres_queue=self.tl_text.res_queue,
                on_success=self.on_started,
                on_failure=self.on_stopped,
                log_cc_errors=print,
                log_ts_errors=print,
                log_tl_errors=print,
            )

        self.control_button.config(text="Start", command=start, state="normal")


if __name__ == "__main__":
    app = IntegratedApp()
    
    def _on_close():
        try:
            app._audio_running = False
            if app._proc is not None:
                app._proc.stop(lambda: app.destroy())
            else:
                app.destroy()
        except Exception:
            app.destroy()

    app.protocol("WM_DELETE_WINDOW", _on_close)

    try:
        app.mainloop()
    except KeyboardInterrupt:
        _on_close()
