"""gallery.py - the Changeling gallery: everything you've made with any Changeling, newest first.

Every Changeling still asks where to save, as before; when it has saved, it also notes the file here (remember()).
The list is one file for all of them, in your home folder: ~/.changeling/gallery.jsonl, with a small picture of each
in ~/.changeling/thumbs/ (so the gallery still shows what it was if the file is moved).

    python gallery.py            the gallery window on its own
    from gallery import remember, show
    remember(path, "Chronos", image=rgb_array)    after saving (the picture is optional: it's read from the file)
    show(root)                                    the gallery window, over an app's own window

Only Python's standard library (and Tk) to show it, so Changeling Studio can open it before anything's installed;
making the small pictures uses Pillow or OpenCV when the app has them. The same file is in every Changeling.
"""
import datetime
import hashlib
import json
import os
import subprocess
import sys

HOME = os.environ.get("CHANGELING_GALLERY") or os.path.join(os.path.expanduser("~"), ".changeling")
LIST = os.path.join(HOME, "gallery.jsonl")
THUMBS = os.path.join(HOME, "thumbs")
THUMB = 220
PICTURES = (".png", ".jpg", ".jpeg", ".webp", ".bmp", ".jfif")
VIDEOS = (".mp4", ".mov", ".avi", ".mkv", ".webm")
SOUNDS = (".wav", ".mp3", ".flac", ".ogg")


# ---- noting what was made
def remember(path, app, image=None):
    """Note a file a Changeling just saved. Never raises: a gallery problem mustn't spoil a save."""
    try:
        path = os.path.abspath(path)
        os.makedirs(THUMBS, exist_ok=True)
        when = datetime.datetime.now().isoformat(timespec="seconds")
        thumb = os.path.join(THUMBS, hashlib.sha1((path + when).encode("utf-8")).hexdigest()[:16] + ".png")
        if not _thumbnail(path, image, thumb):
            thumb = None
        with open(LIST, "a", encoding="utf-8") as f:
            f.write(json.dumps({"time": when, "app": app, "path": path, "thumb": thumb}) + "\n")
    except Exception as e:  # noqa: BLE001
        print(f"Gallery: not noted ({e})")


def _thumbnail(path, image, out):
    """A small PNG of the picture (or a video's first frame) at out; False if there's nothing to show."""
    ext = os.path.splitext(path)[1].lower()
    try:
        from PIL import Image
        if image is not None:
            pic = image if isinstance(image, Image.Image) else Image.fromarray(image)
        elif ext in PICTURES:
            pic = Image.open(path)
        elif ext in VIDEOS:
            import cv2
            cap = cv2.VideoCapture(path)
            cap.set(cv2.CAP_PROP_POS_FRAMES, max(0, int(cap.get(cv2.CAP_PROP_FRAME_COUNT) // 3)))
            ok, frame = cap.read()
            cap.release()
            if not ok:
                return False
            pic = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
        else:
            return False
        pic = pic.convert("RGB")
        pic.thumbnail((THUMB, THUMB))
        pic.save(out)
        return True
    except Exception:  # noqa: BLE001 - no Pillow, or no OpenCV: try OpenCV alone
        try:
            import cv2
            img = None
            if image is not None:
                img = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
            elif ext in PICTURES:
                img = cv2.imread(path)
            if img is None:
                return False
            k = THUMB / max(img.shape[:2])
            if k < 1:
                img = cv2.resize(img, (int(img.shape[1] * k), int(img.shape[0] * k)), interpolation=cv2.INTER_AREA)
            return bool(cv2.imwrite(out, img))
        except Exception:  # noqa: BLE001
            return False


def entries():
    """Everything noted, newest first (a file saved twice shows once, the newest)."""
    if not os.path.exists(LIST):
        return []
    seen, out = set(), []
    with open(LIST, encoding="utf-8") as f:
        lines = [json.loads(l) for l in f if l.strip().startswith("{")]
    for e in reversed(lines):
        if e.get("path") in seen:
            continue
        seen.add(e.get("path"))
        out.append(e)
    return out


def forget(path):
    """Take a file off the gallery (the file itself stays where it is)."""
    keep = [e for e in reversed(entries()) if e.get("path") != path]
    with open(LIST, "w", encoding="utf-8") as f:
        for e in keep:
            f.write(json.dumps(e) + "\n")


def open_file(path):
    if sys.platform.startswith("win"):
        os.startfile(path)  # noqa: S606
    else:
        subprocess.Popen(["open" if sys.platform == "darwin" else "xdg-open", path])


def show_in_folder(path):
    if sys.platform.startswith("win"):
        subprocess.Popen(["explorer", "/select,", os.path.normpath(path)])
    else:
        open_file(os.path.dirname(path))


# ---- the window
GREY, PANEL, LINE, TEXT, MUTED, PINK, DEEP = "#1a1a1a", "#242424", "#3a3a3a", "#e6e6e6", "#9a9a9a", "#e0408a", "#c2186b"


def show(parent=None, app=None):
    """The gallery window. parent: an app's window (it opens on top of it), or None for a window of its own.
    app: start filtered to that Changeling's work."""
    import tkinter as tk

    own = parent is None
    win = tk.Tk() if own else tk.Toplevel(parent)
    win.title("Changeling Gallery")
    win.configure(bg=GREY)
    win.geometry("1050x760")
    if not own:
        win.transient(parent)
        win.lift()
    ico = os.path.join(os.path.dirname(os.path.abspath(__file__)), "logo.ico")
    if os.name == "nt" and os.path.exists(ico):
        try:
            win.iconbitmap(ico)
        except tk.TclError:
            pass
    head = tk.Frame(win, bg=GREY)
    head.pack(fill="x", padx=22, pady=(18, 4))
    tk.Label(head, text="CHANGELING", fg=MUTED, bg=GREY, font=("Segoe UI", 9, "bold")).pack(anchor="w")
    tk.Label(head, text="Gallery", fg=PINK, bg=GREY, font=("Segoe UI", 26, "bold")).pack(anchor="w")
    tk.Label(head, text="everything you've made, newest first: click one to open it", fg=MUTED, bg=GREY,
             font=("Segoe UI", 10)).pack(anchor="w")
    filters = tk.Frame(win, bg=GREY)
    filters.pack(fill="x", padx=20, pady=(6, 4))
    body = tk.Frame(win, bg=GREY)
    body.pack(fill="both", expand=True, padx=16, pady=(4, 12))
    canvas = tk.Canvas(body, bg=GREY, highlightthickness=0)
    bar = tk.Scrollbar(body, orient="vertical", command=canvas.yview)
    grid = tk.Frame(canvas, bg=GREY)
    grid.bind("<Configure>", lambda e: canvas.configure(scrollregion=canvas.bbox("all")))
    canvas.create_window((0, 0), window=grid, anchor="nw")
    canvas.configure(yscrollcommand=bar.set)
    canvas.pack(side="left", fill="both", expand=True)
    bar.pack(side="right", fill="y")
    win.bind("<MouseWheel>", lambda e: canvas.yview_scroll(int(-e.delta / 120), "units"))
    win.bind("<Escape>", lambda e: win.destroy())
    state = {"app": app, "images": []}

    def card(e, row, col):
        exists = os.path.exists(e["path"])
        c = tk.Frame(grid, bg=PANEL, highlightbackground=LINE, highlightthickness=1, width=232, height=318)
        c.grid(row=row, column=col, padx=6, pady=6, sticky="n")
        c.pack_propagate(False)
        pic = None
        if e.get("thumb") and os.path.exists(e["thumb"]):
            try:
                pic = tk.PhotoImage(file=e["thumb"])
                state["images"].append(pic)
            except tk.TclError:
                pic = None
        ext = os.path.splitext(e["path"])[1].lower()
        box = tk.Frame(c, bg="#161616", width=THUMB, height=THUMB)     # the same size with or without a picture
        box.pack(padx=6, pady=(6, 4))
        box.pack_propagate(False)
        shown = tk.Label(box, bg="#161616", cursor="hand2" if exists else "arrow")
        if pic is not None:
            shown.configure(image=pic)
        else:
            shown.configure(text="♪" if ext in SOUNDS else "▶" if ext in VIDEOS else "?", fg=MUTED,
                            font=("Segoe UI", 40))
        shown.pack(fill="both", expand=True)
        if exists:
            shown.bind("<Button-1>", lambda ev, p=e["path"]: open_file(p))
        name = os.path.basename(e["path"])
        tk.Label(c, text=name if len(name) < 32 else name[:29] + "...", fg=TEXT if exists else MUTED, bg=PANEL,
                 font=("Segoe UI", 9, "bold")).pack(anchor="w", padx=8)
        when = e.get("time", "").replace("T", "  ")[:17]
        tk.Label(c, text=f"{e.get('app', '?')}  ·  {when}", fg=MUTED, bg=PANEL, font=("Segoe UI", 8)).pack(anchor="w", padx=8)
        if not exists:
            tk.Label(c, text="moved or deleted", fg=PINK, bg=PANEL, font=("Segoe UI", 8)).pack(anchor="w", padx=8)
        links = tk.Frame(c, bg=PANEL)
        links.pack(anchor="w", padx=8, pady=(2, 6))
        if exists:
            folder = tk.Label(links, text="Show in folder", fg="#f29cc0", bg=PANEL, cursor="hand2", font=("Segoe UI", 8))
            folder.pack(side="left", padx=(0, 12))
            folder.bind("<Button-1>", lambda ev, p=e["path"]: show_in_folder(p))
        drop = tk.Label(links, text="Remove from gallery", fg=MUTED, bg=PANEL, cursor="hand2", font=("Segoe UI", 8))
        drop.pack(side="left")
        drop.bind("<Button-1>", lambda ev, p=e["path"]: (forget(p), fill()))

    def fill():
        for w in grid.winfo_children() + filters.winfo_children():
            w.destroy()
        state["images"].clear()
        all_items = entries()
        apps = sorted({e.get("app", "?") for e in all_items})
        for a in [None] + apps:
            on = state["app"] == a
            b = tk.Button(filters, text=a or "All", bg=DEEP if on else PANEL, fg="white", relief="flat", bd=0,
                          activebackground=PINK, activeforeground="white", padx=12, pady=4, cursor="hand2",
                          font=("Segoe UI", 9, "bold" if on else "normal"),
                          command=lambda a=a: (state.update(app=a), fill()))
            b.pack(side="left", padx=(0, 6))
        items = [e for e in all_items if state["app"] in (None, e.get("app"))]
        if not items:
            tk.Label(grid, text="Nothing here yet. Whatever you save in a Changeling shows up here.", fg=MUTED,
                     bg=GREY, font=("Segoe UI", 11)).grid(row=0, column=0, padx=10, pady=30)
        for i, e in enumerate(items[:400]):
            card(e, i // 4, i % 4)
        canvas.yview_moveto(0)

    fill()
    if own:
        win.mainloop()
    return win


if __name__ == "__main__":
    show(app=sys.argv[1] if len(sys.argv) > 1 else None)
