"""window.py - Changeling CopyCat Realtime's window, in Changeling's look (customtkinter, the Cursed pink).

The picture on the left, the controls on the right. The camera, the swap, the virtual camera and the
recording run on their own thread, so the window never waits for the GPU; the window shows the latest
finished frame about 60 times a second.

Keys still work: S swap, O occlusion, V virtual camera, A voice, P pick a persona, R read it again,
C record / stop, Q or Esc quit.
"""
import os
import threading
import time
import tkinter as tk
import webbrowser

import customtkinter as ctk
import cv2
from PIL import Image, ImageTk

from recorder import Recorder
from voice import PRESETS, SR

HERE = os.path.dirname(os.path.abspath(__file__))
NAME = "Changeling CopyCat Realtime"
GITHUB = "https://github.com/CursedPrograms/Changeling-CopyCat-Realtime"
KOFI = "https://ko-fi.com/cursedentertainment"
PINK, RED = "#e0408a", "#d43c3c"



def set_logo(root, here):
    """The Changeling logo as the window's icon (title bar and taskbar). customtkinter puts its own icon in
    about 200 ms after the window opens, so this goes in just after it."""
    ico, png = os.path.join(here, "logo.ico"), os.path.join(here, "logo.png")

    def apply():
        try:
            if os.name == "nt" and os.path.exists(ico):
                root.iconbitmap(ico)
            elif os.path.exists(png):
                root._logo_icon = tk.PhotoImage(file=png)
                root.iconphoto(True, root._logo_icon)
        except tk.TclError:
            pass
    root.after(300, apply)


def logo_image(here, size=56):
    """The logo for the sidebar, or None."""
    png = os.path.join(here, "logo.png")
    if not os.path.exists(png):
        return None
    return ctk.CTkImage(Image.open(png), size=(size, size))

class Window:
    def __init__(self, args, read, still, sw, persona, *, load_persona, pick_persona, open_virtual, voice):
        self.args, self.read, self.still, self.sw = args, read, still, sw
        self.persona = persona
        self.load_persona, self.pick_persona, self.open_virtual = load_persona, pick_persona, open_virtual
        self.voice = voice
        self.recorder = Recorder(os.path.join(HERE, "recordings"))
        if voice is not None:
            voice.taps.append(self.recorder.sound)
        self.swap_on = True
        self.want_virtual = not args.no_virtual
        self.virtual = None
        self.strength = 1.0
        self.want_persona = None          # a path for the worker to load (it owns the models)
        self.latest = None                # (frame, found) from the worker
        self.fps = 0.0
        self.running = True
        self.lock = threading.Lock()

        ctk.set_appearance_mode("dark")
        ctk.set_default_color_theme(os.path.join(HERE, "theme.json"))
        self.root = ctk.CTk()
        set_logo(self.root, HERE)
        self.root.title(NAME)
        self.root.geometry("1280x800")
        self.root.minsize(980, 620)
        self.root.protocol("WM_DELETE_WINDOW", self.quit)
        self.root.grid_columnconfigure(0, weight=1)
        self.root.grid_rowconfigure(0, weight=1)
        self._build()
        self._keys()

    # ---- the window
    def _build(self):
        bg = self.root.cget("fg_color")[1]
        stage = ctk.CTkFrame(self.root, corner_radius=12)
        stage.grid(row=0, column=0, sticky="nsew", padx=(14, 7), pady=14)
        self.view = tk.Label(stage, bg="#16161a", bd=0, highlightthickness=0)
        self.view.pack(fill="both", expand=True, padx=6, pady=6)
        self.view_size = (960, 540)
        self.view.bind("<Configure>", lambda e: setattr(self, "view_size", (max(e.width, 64), max(e.height, 36))))

        side = ctk.CTkScrollableFrame(self.root, width=300, corner_radius=12)
        side.grid(row=0, column=1, sticky="ns", padx=(7, 14), pady=14)
        logo = logo_image(HERE)
        if logo is not None:
            ctk.CTkLabel(side, image=logo, text="").pack(anchor="w", padx=12, pady=(12, 0))
        ctk.CTkLabel(side, text="CHANGELING", text_color="gray55",
                     font=ctk.CTkFont(size=11, weight="bold")).pack(anchor="w", padx=12, pady=(10, 0))
        ctk.CTkLabel(side, text="CopyCat Realtime", font=ctk.CTkFont(size=24, weight="bold"),
                     text_color=PINK).pack(anchor="w", padx=12)
        ctk.CTkLabel(side, text="your expressions, someone else's face", text_color="gray60").pack(anchor="w", padx=12)

        self._section(side, "Persona")
        self.thumb = ctk.CTkLabel(side, text="", width=96, height=96, fg_color=("gray85", "gray17"), corner_radius=10)
        self.thumb.pack(padx=12, pady=(4, 2))
        self.persona_name = ctk.CTkLabel(side, text="", text_color="gray70")
        self.persona_name.pack()
        row = ctk.CTkFrame(side, fg_color="transparent")
        row.pack(fill="x", padx=12, pady=6)
        ctk.CTkButton(row, text="Pick persona  (P)", command=self.pick).pack(side="left", expand=True, fill="x", padx=(0, 4))
        ctk.CTkButton(row, text="Reload", width=70, command=self.reload).pack(side="left")
        self._show_persona()

        self._section(side, "Picture")
        self.swap_sw = self._switch(side, "Swap  (S)", True, lambda v: setattr(self, "swap_on", v))
        occl = self._switch(side, "Occlusion  (O)", self.sw.occlusion_on, lambda v: setattr(self.sw, "occlusion_on", v))
        if self.sw.occluder is None:
            occl.configure(state="disabled", text="Occlusion: no model")
        self.occl_sw = occl
        self.virt_sw = self._switch(side, "Virtual camera  (V)", self.want_virtual, self.set_virtual)
        self.smooth = self._slider(side, "Smooth", 0, 90, 50, lambda v: setattr(self.sw, "smooth", v / 100))
        self._slider(side, "Strength", 0, 100, 100, lambda v: setattr(self, "strength", v / 100))
        self._slider(side, "Persona's skin", 0, 100, int(round(self.sw.skin * 100)), lambda v: setattr(self.sw, "skin", v / 100))

        self._section(side, "Voice  (A)")
        self.voice_pick = ctk.CTkSegmentedButton(side, values=PRESETS, command=self.set_voice)
        self.voice_pick.pack(fill="x", padx=12, pady=4)
        self.echo_sw = self._switch(side, "Echo: trained voices  (E)", bool(self.voice and self.voice.echo), self.set_echo)
        self.voice_note = ctk.CTkLabel(side, text="", text_color="gray55", wraplength=270, justify="left")
        self.voice_note.pack(anchor="w", padx=12)
        self.cable_note = ""
        if self.voice is None:
            self.voice_pick.configure(state="disabled")
            self.echo_sw.configure(state="disabled")
            self.voice_note.configure(text="No microphone found.")
        else:
            self.voice_pick.set(self.voice.preset)
            if self.voice.echo_dir is None:
                self.echo_sw.configure(state="disabled", text="Echo: not installed")
            if not self.voice.to_call:
                self.cable_note = "No VB-Audio Virtual Cable: the voice goes into recordings only."
            self.root.after(500, self._voice_status)

        self._section(side, "Record  (C)")
        self.rec_btn = ctk.CTkButton(side, text="●  Record", height=42, font=ctk.CTkFont(size=15, weight="bold"),
                                     command=self.toggle_record)
        self.rec_btn.pack(fill="x", padx=12, pady=4)
        self.rec_note = ctk.CTkLabel(side, text="Picture and voice, to recordings/", text_color="gray55",
                                     cursor="hand2", wraplength=270, justify="left")
        self.rec_note.pack(anchor="w", padx=12)
        self.rec_note.bind("<Button-1>", lambda e: self.open_recordings())

        self.status = ctk.CTkLabel(side, text="", text_color="gray60", justify="left", wraplength=270)
        self.status.pack(anchor="w", padx=12, pady=(10, 2))
        links = ctk.CTkFrame(side, fg_color="transparent")
        links.pack(fill="x", padx=12, pady=(8, 2))
        for text, url in (("GitHub", GITHUB), ("Ko-fi", KOFI)):
            l = ctk.CTkLabel(links, text=text, text_color=("#c2186b", "#f29cc0"), cursor="hand2")
            l.pack(side="left", padx=(0, 14))
            l.bind("<Button-1>", lambda e, u=url: webbrowser.open(u))
        ctk.CTkLabel(side, text="© Cursed Entertainment 2026", text_color="gray45").pack(anchor="w", padx=12, pady=(0, 10))

    def _section(self, parent, text):
        ctk.CTkLabel(parent, text=text.upper(), text_color="gray50",
                     font=ctk.CTkFont(size=11, weight="bold")).pack(anchor="w", padx=12, pady=(10, 1))

    def _switch(self, parent, text, on, command):
        var = ctk.BooleanVar(value=on)
        sw = ctk.CTkSwitch(parent, text=text, variable=var, command=lambda: command(var.get()))
        sw.var = var
        sw.pack(anchor="w", padx=12, pady=3)
        return sw

    def _slider(self, parent, text, lo, hi, value, command):
        row = ctk.CTkFrame(parent, fg_color="transparent")
        row.pack(fill="x", padx=12, pady=(6, 0))
        ctk.CTkLabel(row, text=text).pack(side="left")
        shown = ctk.CTkLabel(row, text=f"{value}%", text_color="gray60")
        shown.pack(side="right")

        def moved(v):
            shown.configure(text=f"{int(v)}%")
            command(int(v))
        s = ctk.CTkSlider(parent, from_=lo, to=hi, number_of_steps=hi - lo, command=moved)
        s.set(value)
        s.pack(fill="x", padx=12, pady=(0, 4))
        command(value)
        return s

    def _keys(self):
        def flip(switch):
            if switch.cget("state") != "disabled":
                switch.toggle()
        keys = {"s": lambda: flip(self.swap_sw), "o": lambda: flip(self.occl_sw), "v": lambda: flip(self.virt_sw),
                "a": self.next_voice, "e": lambda: flip(self.echo_sw), "p": self.pick, "r": self.reload, "c": self.toggle_record, "q": self.quit}
        for k, f in keys.items():
            self.root.bind(f"<KeyPress-{k}>", lambda e, f=f: f())
        self.root.bind("<Escape>", lambda e: self.quit())

    # ---- what the controls do
    def _show_persona(self):
        if not self.persona:
            self.thumb.configure(image=None, text="no persona")
            self.persona_name.configure(text="Pick one, or put her in personas/input.jpeg")
            return
        img = cv2.imread(self.persona_path()) if self.persona_path() else None
        if img is not None:
            h, w = img.shape[:2]
            s = min(h, w)
            crop = img[(h - s) // 2:(h + s) // 2, (w - s) // 2:(w + s) // 2]
            pil = Image.fromarray(cv2.cvtColor(crop, cv2.COLOR_BGR2RGB))
            self.thumb.configure(image=ctk.CTkImage(pil, size=(96, 96)), text="")
        self.persona_name.configure(text=self.persona[0])

    def persona_path(self):
        return self.args.persona if self.args.persona and os.path.exists(self.args.persona) else None

    def pick(self):
        path = self.pick_persona()
        if path:
            self.args.persona = path
            self.want_persona = path

    def reload(self):
        self.want_persona = self.args.persona

    def set_virtual(self, on):
        self.want_virtual = on

    def set_voice(self, preset):
        if self.voice is not None:
            self.voice.preset = preset

    def set_echo(self, on):
        if self.voice is not None:
            self.voice.echo = on and self.voice.echo_dir is not None

    def _voice_status(self):
        """Echo's state (loading, failed) or the cable note, under the voices."""
        if self.voice is None:
            return
        self.voice_note.configure(text=self.voice.echo_state() or self.cable_note)
        self.root.after(500, self._voice_status)

    def next_voice(self):
        if self.voice is not None:
            self.voice_pick.set(self.voice.next_preset())

    def toggle_record(self):
        if self.recorder.active:
            self.rec_btn.configure(text="Saving ...", state="disabled")
            self.root.update_idletasks()
            path, message = self.recorder.stop()
            self.rec_btn.configure(text="●  Record", fg_color=PINK, state="normal")
            self.rec_note.configure(text=f"{message}: {os.path.basename(path)}  (click: open the folder)" if path else message)
            return
        latest = self.latest
        if latest is None:
            return
        h, w = latest[0].shape[:2]
        self.recorder.start((w, h), SR if (self.voice is not None and self.voice.ready) else None)
        self.rec_btn.configure(fg_color=RED, hover_color="#a82e2e")

    def open_recordings(self):
        folder = self.recorder.folder
        os.makedirs(folder, exist_ok=True)
        os.startfile(folder) if hasattr(os, "startfile") else webbrowser.open(folder)

    def quit(self):
        self.running = False
        if self.recorder.active:
            self.recorder.stop()
        self.root.after(150, self.root.destroy)

    # ---- the worker: camera, swap, virtual camera, recording
    def work(self):
        t_last = time.monotonic()
        while self.running:
            if self.want_persona:
                path, self.want_persona = self.want_persona, None
                self.persona = self.load_persona(path, self.sw) or self.persona
                self.root.after(0, self._show_persona)
            frame = self.read()
            if frame is None:
                self.root.after(0, lambda: self.status.configure(text="The camera stopped giving pictures."))
                break
            if self.virtual is None and self.want_virtual:
                self.virtual = self.open_virtual(frame.shape[1], frame.shape[0], self.args.fps)
                if self.virtual is None:
                    self.want_virtual = False
                    self.root.after(0, lambda: (self.virt_sw.deselect(), self.virt_sw.configure(text="Virtual camera: no OBS")))
            elif self.virtual is not None and not self.want_virtual:
                self.virtual.close()
                self.virtual = None
            out, found = frame, False
            if self.swap_on and self.persona:
                out, found = self.sw.swap(frame, self.persona[1], self.strength)
            if self.virtual is not None:
                self.virtual.send(out)
            self.recorder.frame(out)
            now = time.monotonic()
            self.fps = self.fps * 0.9 + 0.1 / max(now - t_last, 1e-3)
            t_last = now
            self.latest = (out, found)
            if self.still:
                time.sleep(1 / 30)
        if self.virtual is not None:
            self.virtual.close()

    def show(self):
        latest = self.latest
        if latest is not None and self.running:
            frame, found = latest
            vw, vh = self.view_size
            h, w = frame.shape[:2]
            k = min(vw / w, vh / h)
            small = cv2.resize(frame, (max(int(w * k), 1), max(int(h * k), 1)), interpolation=cv2.INTER_AREA)
            if self.recorder.active:
                cv2.circle(small, (22, 22), 8, (60, 60, 212), -1)
            photo = ImageTk.PhotoImage(Image.fromarray(cv2.cvtColor(small, cv2.COLOR_BGR2RGB)))
            self.view.configure(image=photo)
            self.view.image = photo
            lines = [f"{self.fps:4.1f} fps", "face found" if found else ("no face in view" if self.swap_on else "swap off")]
            if self.virtual is not None:
                lines.append("to the call: OBS Virtual Camera")
            self.status.configure(text="   ·   ".join(lines))
            if self.recorder.active:
                m, s = divmod(int(self.recorder.seconds), 60)
                self.rec_btn.configure(text=f"■  Stop   {m}:{s:02d}")
        if self.running:
            self.root.after(16, self.show)

    def run(self):
        threading.Thread(target=self.work, daemon=True, name="camera").start()
        self.root.after(50, self.show)
        self.root.mainloop()
        self.running = False
