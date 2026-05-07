import tkinter as tk
from tkinter import ttk, filedialog, messagebox
import threading
import numpy as np
import librosa
import sounddevice as sd
import soundfile as sf
import os
from spectrum import show_spectrum

from effects.delay import apply_delay
from effects.filters import lowpass_filter, highpass_filter, bandpass_filter
from effects.reverb import apply_reverb, generate_impulse_response
from effects.chorus import apply_chorus
from effects.distortion import apply_distortion

# Effect definitions
def apply_volume(audio, gain=1.0):
    output = gain * audio
    return np.clip(output, -1.0, 1.0)
# each entry describes one pedal's parameters
EFFECT_DEFS = {
    "Delay": {
        "fn": lambda audio, sr, p: apply_delay(audio, sr, delay_sec=p["Time (s)"], alpha=p["Feedback"]),
        "params": {
            "Time (s)":  {"min": 0.1,  "max": 1.0,   "default": 0.3,  "resolution": 0.01},
            "Feedback":  {"min": 0.0,  "max": 1.0,   "default": 0.5,  "resolution": 0.01},
        },
    },
    "Reverb": {
        "fn": lambda audio, sr, p: apply_reverb(audio, generate_impulse_response(sr, duration=p["Size (s)"], decay=p["Decay"])),
        "params": {
            "Size (s)": {"min": 0.05, "max": 1.5,   "default": 0.3,  "resolution": 0.01},
            "Decay":    {"min": 1.0,  "max": 15.0,  "default": 5.0,  "resolution": 0.1},
        },
    },
    "Low Pass": {
        "fn": lambda audio, sr, p: lowpass_filter(audio, sr, cutoff=p["Cutoff (Hz)"], order=int(p["Order"])),
        "params": {
            "Cutoff (Hz)": {"min": 200,  "max": 8000, "default": 1000, "resolution": 10},
            "Order":       {"min": 1,    "max": 8,    "default": 4,    "resolution": 1},
        },
    },
    "High Pass": {
        "fn": lambda audio, sr, p: highpass_filter(audio, sr, cutoff=p["Cutoff (Hz)"], order=int(p["Order"])),
        "params": {
            "Cutoff (Hz)": {"min": 200,  "max": 8000, "default": 1000, "resolution": 10},
            "Order":       {"min": 1,    "max": 8,    "default": 4,    "resolution": 1},
        },
    },
    "Band Pass": {
        "fn": lambda audio, sr, p: bandpass_filter(audio, sr, lowcut=p["Low (Hz)"], highcut=p["High (Hz)"], order=int(p["Order"])),
        "params": {
            "Low (Hz)":  {"min": 100,  "max": 4000,  "default": 500,  "resolution": 10},
            "High (Hz)": {"min": 500,  "max": 10000, "default": 2000, "resolution": 10},
            "Order":     {"min": 1,    "max": 8,     "default": 4,    "resolution": 1},
        },
    },
    "Distortion": {
        "fn": lambda audio, sr, p: apply_distortion(audio, gain=p["Gain"], mix=p["Mix"]),
        "params": {
            "Gain": {"min": 1.0,  "max": 20.0, "default": 2.0,  "resolution": 0.1},
            "Mix":  {"min": 0.0,  "max": 1.0,  "default": 1.0,  "resolution": 0.01},
        },
    },
    "Chorus": {
        "fn": lambda audio, sr, p: apply_chorus(audio, sr, depth_ms=p["Depth (ms)"], rate_hz=p["Rate (Hz)"], base_delay_ms=p["Base Delay (ms)"], mix=p["Mix"]),
        "params": {
            "Depth (ms)":       {"min": 1.0,  "max": 20.0,  "default": 8.0,  "resolution": 0.5},
            "Rate (Hz)":        {"min": 0.1,  "max": 5.0,   "default": 1.5,  "resolution": 0.1},
            "Base Delay (ms)":  {"min": 5.0,  "max": 50.0,  "default": 15.0, "resolution": 0.5},
            "Mix":              {"min": 0.0,  "max": 1.0,   "default": 0.5,  "resolution": 0.01},
        },
    },
}

# Colours
BG       = "#ffffff"
BOARD    = "#ffffff"
PANEL    = "#b3f2ff"
ACCENT   = "#307130"
ACCENT2  = "#0b4e1d"
TEXT     = "#000000"
MUTED    = "#414ce8"
BTN_BG   = "#7bef7f"
BTN_ACT  = "#b4ffad"
BYPASS   = "#8b2020"
ACTIVE   = "#2a6a2a"
MONO     = ("Courier", 10)
MONO_SM  = ("Courier", 9)


# Main application
class PedalboardApp(tk.Tk):
    def __init__(self):
        super().__init__()
        self.title("Pedalboard")
        self.configure(bg=BG)
        self.resizable(True, True)
        self.minsize(700, 600)

        self.audio = None
        self.sample_rate = None
        self.file_path = None
        
        self.master_volume = tk.DoubleVar(value=1.0)

        # chain: list of {"name": str, "enabled": bool, "vars": {param: DoubleVar}}
        self.chain = []

        self._build_ui()
        
    def _build_ui(self):
        # Top bar: file loading 
        top = tk.Frame(self, bg=BG, pady=8, padx=12)
        top.pack(fill="x")

        tk.Label(top, text="◆  PEDALBOARD", font=("Courier", 13, "bold"),
                 fg=ACCENT, bg=BG).pack(side="left", padx=(0, 20))

        self.file_lbl = tk.Label(top, text="No file loaded",
                                 font=MONO_SM, fg=MUTED, bg=BG)
        self.file_lbl.pack(side="left", padx=6)

        self._btn(top, "Browse Audio", self._load_file).pack(side="left", padx=4)
        tk.Label(top, text="Volume", font=MONO_SM, fg=TEXT, bg=BG).pack(side="left", padx=(12, 4))

        self.volume_lbl = tk.Label(top, text="1.00x", font=MONO_SM, fg=ACCENT, bg=BG, width=6)
        self.volume_lbl.pack(side="left")

        def update_volume_label(val):
            self.volume_lbl.config(text=f"{float(val):.2f}x")

        volume_slider = ttk.Scale(
            top,
            from_=0.0,
            to=2.0,
            variable=self.master_volume,
            orient="horizontal",
            command=update_volume_label,
            length=120
        )
        volume_slider.pack(side="left", padx=4)
        
        self._btn(top, "Process & Export", self._process, accent=True).pack(side="right", padx=4)
        self._btn(top, "Preview", self._preview).pack(side="right", padx=4)
        self._btn(top, "Stop", self._stop_preview).pack(side="right", padx=4)
        self._btn(top, "Show Spectrum", self._show_spectrum).pack(side="right", padx=4)

        self.status_lbl = tk.Label(top, text="", font=MONO_SM, fg=ACCENT, bg=BG)
        self.status_lbl.pack(side="right", padx=8)

        sep = tk.Frame(self, bg=ACCENT2, height=1)
        sep.pack(fill="x")

        # Main area: chain (left) + add-pedal panel (right)
        main = tk.Frame(self, bg=BG)
        main.pack(fill="both", expand=True, padx=12, pady=10)
        main.columnconfigure(0, weight=1)
        main.columnconfigure(1, weight=0)
        main.rowconfigure(0, weight=1)

        # Left: signal chain
        chain_outer = tk.Frame(main, bg=BOARD, bd=0, relief="flat")
        chain_outer.grid(row=0, column=0, sticky="nsew", padx=(0, 10))

        chain_header = tk.Frame(chain_outer, bg=BOARD)
        chain_header.pack(fill="x", padx=10, pady=(10, 4))
        tk.Label(chain_header, text="SIGNAL CHAIN",
                 font=("Courier", 9), fg=MUTED, bg=BOARD).pack(side="left")
        tk.Label(chain_header, text="↑ ↓ to reorder  ·  X to remove",
                 font=("Courier", 8), fg=MUTED, bg=BOARD).pack(side="right")

        # Scrollable chain list
        chain_frame_wrap = tk.Frame(chain_outer, bg=BOARD)
        chain_frame_wrap.pack(fill="both", expand=True, padx=6, pady=(0, 8))

        self.chain_canvas = tk.Canvas(chain_frame_wrap, bg=BOARD, highlightthickness=0)
        scrollbar = ttk.Scrollbar(chain_frame_wrap, orient="vertical",
                                  command=self.chain_canvas.yview)
        self.chain_canvas.configure(yscrollcommand=scrollbar.set)
        scrollbar.pack(side="right", fill="y")
        self.chain_canvas.pack(side="left", fill="both", expand=True)

        self.chain_inner = tk.Frame(self.chain_canvas, bg=BOARD)
        self.chain_window = self.chain_canvas.create_window(
            (0, 0), window=self.chain_inner, anchor="nw")
        self.chain_inner.bind("<Configure>", self._on_chain_configure)
        self.chain_canvas.bind("<Configure>", self._on_canvas_configure)

        # Empty label shown when chain is empty
        self.empty_lbl = tk.Label(
            self.chain_inner,
            text="\n  Add effects from the panel →\n  to build your signal chain.\n",
            font=MONO_SM, fg=MUTED, bg=BOARD)
        self.empty_lbl.pack(pady=20)

        # Right: add-pedal panel
        right = tk.Frame(main, bg=PANEL, width=180)
        right.grid(row=0, column=1, sticky="ns")
        right.pack_propagate(False)

        tk.Label(right, text="EFFECTS", font=("Courier", 9),
                 fg=MUTED, bg=PANEL).pack(pady=(10, 6))

        for name in EFFECT_DEFS:
            self._btn(right, name,
                      lambda n=name: self._add_pedal(n),
                      width=18).pack(pady=3, padx=10)

    # Widget helpers
    def _btn(self, parent, text, cmd, accent=False, width=None):
        kw = dict(text=text, command=cmd, font=MONO_SM, cursor="hand2",
                  bg=BTN_ACT if accent else BTN_BG,
                  fg=ACCENT if accent else TEXT,
                  activebackground=BTN_ACT, activeforeground=ACCENT,
                  relief="flat", bd=0, padx=10, pady=4)
        if width:
            kw["width"] = width
        b = tk.Button(parent, **kw)
        b.bind("<Enter>", lambda e, b=b: b.config(bg=BTN_ACT))
        b.bind("<Leave>", lambda e, b=b: b.config(bg=BTN_ACT if accent else BTN_BG))
        return b

    # Canvas resize helpers
    def _on_chain_configure(self, event):
        self.chain_canvas.configure(scrollregion=self.chain_canvas.bbox("all"))

    def _on_canvas_configure(self, event):
        self.chain_canvas.itemconfig(self.chain_window, width=event.width)

    # File loading
    def _load_file(self):
        path = filedialog.askopenfilename(
            initialdir= os.getcwd(),
            title="Select audio file",
            filetypes=[("Audio files", "*.mp3 *.wav *.flac *.ogg"), ("All files", "*.*")])
        if not path:
            return
        self.file_path = path
        self.file_lbl.config(text=os.path.basename(path), fg=ACCENT)
        self._set_status("Loading…")
        threading.Thread(target=self._load_audio, daemon=True).start()

    def _load_audio(self):
        try:
            audio, sr = librosa.load(self.file_path, sr=None, mono=False)
            if audio.ndim == 2:
                audio = audio.T
            mx = np.max(np.abs(audio))
            if mx > 0:
                audio = audio / mx
            self.audio = audio
            self.sample_rate = sr
            self._set_status(f"Loaded  {sr} Hz")
        except Exception as e:
            self._set_status("Load error")
            messagebox.showerror("Load error", str(e))

    # Chain management
    def _add_pedal(self, name):
        entry = {
            "name": name,
            "enabled": tk.BooleanVar(value=True),
            "vars": {
                pname: tk.DoubleVar(value=pdef["default"])
                for pname, pdef in EFFECT_DEFS[name]["params"].items()
            },
        }
        self.chain.append(entry)
        self._rebuild_chain_ui()

    def _remove_pedal(self, idx):
        self.chain.pop(idx)
        self._rebuild_chain_ui()

    def _move_pedal(self, idx, direction):
        new_idx = idx + direction
        if 0 <= new_idx < len(self.chain):
            self.chain[idx], self.chain[new_idx] = self.chain[new_idx], self.chain[idx]
            self._rebuild_chain_ui()

    # Rebuild the chain UI
    def _rebuild_chain_ui(self):
        for w in self.chain_inner.winfo_children():
            w.destroy()

        if not self.chain:
            self.empty_lbl = tk.Label(
                self.chain_inner,
                text="\n  Add effects from the panel →\n  to build your signal chain.\n",
                font=MONO_SM, fg=MUTED, bg=BOARD)
            self.empty_lbl.pack(pady=20)
            return

        for i, entry in enumerate(self.chain):
            self._build_pedal_card(self.chain_inner, entry, i)

        # Signal path display at the bottom
        sp_frame = tk.Frame(self.chain_inner, bg=BOARD)
        sp_frame.pack(fill="x", padx=10, pady=(6, 4))
        tk.Label(sp_frame, text="IN", font=MONO_SM, fg=ACCENT, bg=BOARD).pack(side="left")
        for e in self.chain:
            tk.Label(sp_frame, text=" → ", font=MONO_SM, fg=MUTED, bg=BOARD).pack(side="left")
            col = ACCENT if e["enabled"].get() else MUTED
            tk.Label(sp_frame, text=e["name"].upper(), font=MONO_SM,
                     fg=col, bg=BOARD).pack(side="left")
        tk.Label(sp_frame, text=" → OUT", font=MONO_SM, fg=ACCENT, bg=BOARD).pack(side="left")

    def _build_pedal_card(self, parent, entry, idx):
        name = entry["name"]
        params = EFFECT_DEFS[name]["params"]

        # Card frame
        card = tk.Frame(parent, bg=PANEL, relief="flat", bd=0)
        card.pack(fill="x", padx=8, pady=5)

        # Inner border effect using a thin accent strip on the left
        strip = tk.Frame(card, bg=ACCENT2, width=3)
        strip.pack(side="left", fill="y")

        body = tk.Frame(card, bg=PANEL)
        body.pack(side="left", fill="both", expand=True, padx=8, pady=6)

        # Header row
        hdr = tk.Frame(body, bg=PANEL)
        hdr.pack(fill="x", pady=(0, 4))

        # Bypass toggle
        def make_bypass_toggle(e=entry, s=strip):
            def toggle():
                col = ACCENT2 if e["enabled"].get() else "#4a2020"
                s.config(bg=col)
                self._rebuild_chain_ui()
            return toggle

        byp = tk.Checkbutton(hdr, variable=entry["enabled"],
                             command=make_bypass_toggle(),
                             bg=PANEL, activebackground=PANEL,
                             selectcolor=PANEL, fg=ACTIVE,
                             text="ON", font=MONO_SM,
                             cursor="hand2")
        byp.pack(side="left")

        tk.Label(hdr, text=name.upper(), font=("Courier", 11, "bold"),
                 fg=ACCENT, bg=PANEL).pack(side="left", padx=8)

        # Up / Down / Remove buttons
        ctrl = tk.Frame(hdr, bg=PANEL)
        ctrl.pack(side="right")
        self._btn(ctrl, "↑", lambda i=idx: self._move_pedal(i, -1)).pack(side="left", padx=2)
        self._btn(ctrl, "↓", lambda i=idx: self._move_pedal(i,  1)).pack(side="left", padx=2)
        rm = tk.Button(ctrl, text="X", font=MONO_SM, fg="#cc4444", bg=PANEL,
                       activebackground=PANEL, activeforeground="#ff6666",
                       relief="flat", bd=0, padx=6, cursor="hand2",
                       command=lambda i=idx: self._remove_pedal(i))
        rm.pack(side="left", padx=2)

        # Parameter sliders
        for pname, pdef in params.items():
            var = entry["vars"][pname]
            self._slider_row(body, pname, var, pdef)

    def _slider_row(self, parent, label, var, pdef):
        row = tk.Frame(parent, bg=PANEL)
        row.pack(fill="x", pady=2)

        tk.Label(row, text=label, font=MONO_SM, fg=TEXT, bg=PANEL,
                 width=18, anchor="w").pack(side="left")

        val_lbl = tk.Label(row, text=self._fmt(var.get(), pdef),
                           font=MONO_SM, fg=ACCENT, bg=PANEL, width=8, anchor="e")
        val_lbl.pack(side="right")

        def on_slide(val, lbl=val_lbl, v=var, d=pdef):
            lbl.config(text=self._fmt(float(val), d))

        sl = ttk.Scale(row, from_=pdef["min"], to=pdef["max"],
                       variable=var, orient="horizontal",
                       command=on_slide)
        sl.pack(side="left", fill="x", expand=True, padx=6)

    @staticmethod
    def _fmt(val, pdef):
        res = pdef["resolution"]
        if res >= 1:
            return str(int(round(val)))
        decimals = len(str(res).rstrip("0").split(".")[-1]) if "." in str(res) else 0
        return f"{val:.{decimals}f}"

    # Processing
    def _process(self):
        if self.audio is None:
            messagebox.showwarning("No audio", "Please load an audio file first.")
            return
        if not self.chain:
            messagebox.showwarning("Empty chain", "Add at least one effect to the chain.")
            return

        out_path = filedialog.asksaveasfilename(
            title="Save processed audio",
            defaultextension=".wav",
            filetypes=[("WAV file", "*.wav")])
        if not out_path:
            return

        self._set_status("Processing…")
        threading.Thread(target=self._run_chain, args=(out_path,), daemon=True).start()

    def _show_spectrum(self):
        if self.audio is None:
            messagebox.showwarning("No audio", "Please load an audio file first.")
            return
        if not self.chain:
            messagebox.showwarning("Empty chain", "Add at least one effect to the chain.")
            return

        self._set_status("Computing spectrum…")

        def worker():
            try:
                audio_after, sr = self._process_chain_to_memory()
                # Show spectrum on the main thread
                self.after(0, lambda: show_spectrum(
                    self.audio, audio_after, sr,
                    title="Frequency Spectrum — Before vs After"
                ))
                self._set_status("Spectrum ready.")
            except Exception as e:
                self._set_status("Spectrum error.")
                messagebox.showerror("Spectrum error", str(e))

        threading.Thread(target=worker, daemon=True).start()
        
    def _run_chain(self, out_path):
        try:
            audio, sr = self._process_chain_to_memory()
            sf.write(out_path, audio, sr)
            self._set_status(f"Saved: {os.path.basename(out_path)}")
            messagebox.showinfo("Done", f"Saved to:\n{out_path}")
        except Exception as e:
            self._set_status("Error")
            messagebox.showerror("Processing error", str(e))

    def _set_status(self, msg):
        # Safe to call from background threads
        self.after(0, lambda: self.status_lbl.config(text=msg))
        
    def _process_chain_to_memory(self):
        audio = self.audio.copy()
        sr = self.sample_rate

        for entry in self.chain:
            if not entry["enabled"].get():
                continue

            name = entry["name"]
            params = {pname: var.get() for pname, var in entry["vars"].items()}
            fn = EFFECT_DEFS[name]["fn"]

            audio = fn(audio, sr, params)

            mx = np.max(np.abs(audio))
            if mx > 1.0:
                audio = audio / mx

        audio = audio * self.master_volume.get()
        audio = np.clip(audio, -1.0, 1.0)
        return audio, sr


    def _preview(self):
        if self.audio is None:
            messagebox.showwarning("No audio", "Please load an audio file first.")
            return

        if not self.chain:
            messagebox.showwarning("Empty chain", "Add at least one effect to preview.")
            return

        self._set_status("Previewing...")

        def worker():
            try:
                audio, sr = self._process_chain_to_memory()
                sd.stop()
                sd.play(audio, sr)
                self._set_status("Playing preview")
            except Exception as e:
                self._set_status("Preview error")
                messagebox.showerror("Preview error", str(e))

        threading.Thread(target=worker, daemon=True).start()


    def _stop_preview(self):
        sd.stop()
        self._set_status("Preview stopped")


if __name__ == "__main__":
    app = PedalboardApp()
    app.geometry("900x620")
    app.mainloop()
