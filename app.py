"""app.py - Persona Cam: a live face swap for video calls. You become the face in personas/input.jpeg,
with your own expressions (your smile, blinks, mouth as you talk), sent to the call as a camera.

    run.bat                      the webcam, a preview window, and OBS's virtual camera for your calls
    run.bat --camera 1           another webcam
    run.bat --source clip.mp4    a video or a picture instead of the webcam (for trying it out)
    run.bat --no-virtual         just the preview, no virtual camera
    run.bat --persona other.jpg  become another face for this run (instead of personas/input.jpeg)

In your call (Zoom, Teams, Discord, Meet...) pick the camera "OBS Virtual Camera". OBS Studio must be
installed (its virtual camera driver), but doesn't have to be running.

The face you become: personas/input.jpeg, a generated person who doesn't exist (see README.md). Replace
the file and press R: she changes without a restart.

Keys in the preview window:
    S          swap on / off (your own face, for a moment)
    O          occlusion on / off (hands, hair, a cup in front of the face stay in front of hers)
    V          virtual camera on / off
    R          read personas/input.jpeg again (after replacing it)
    Q or Esc   quit
Sliders: smooth (steadies where the face sits, frame to frame), strength (how much of the swapped face
shows, 100 = all of it).

The swap: InsightFace finds the face (SCRFD, buffalo_l) and inswapper_128 draws the persona's identity
onto it, on the GPU. The models: buffalo_l in ~/.insightface (downloaded the first time), inswapper_128
in models/ (download it once: see README.md). InsightFace's models are for non-commercial use.
"""
import argparse
import os
import sys
import time

import torch  # noqa: F401  first: its CUDA 12 / cuDNN DLLs are the ones onnxruntime-gpu runs on

import cv2
import numpy as np
import insightface
from insightface.app.common import Face
import mediapipe as mp

HERE = os.path.dirname(os.path.abspath(__file__))
WINDOW = "Persona Cam"
SWAP_MODEL = os.path.join(HERE, "models", "inswapper_128.onnx")
# FaceFusion's occluder (XSeg): which part of the face is really face, and which is a hand, hair or a cup
# in front of it, so the swap leaves those alone. Optional: without it the swap paints over them.
OCCLUDER_MODEL = os.path.join(HERE, "models", "xseg_1.onnx")
# InsightFace's buffalo_l: only the two of its five models Persona Cam uses (FaceAnalysis loads all five,
# twice over for two of them: that was most of a 100 s start)
BUFFALO = os.path.join(os.path.expanduser("~"), ".insightface", "models", "buffalo_l")
PERSONA = os.path.join(HERE, "personas", "input.jpeg")            # hers: private, not in git
PLACEHOLDER = os.path.join(HERE, "personas", "placeholder.jpeg")  # the repo's, while input.jpeg isn't there
# The GPU first, CPU if it can't. HEURISTIC: cuDNN picks its way of running each layer straight away; the
# default (EXHAUSTIVE) times every way there is first, which made the first start take nearly 2 minutes.
PROVIDERS = [("CUDAExecutionProvider", {"cudnn_conv_algo_search": "HEURISTIC"}), "CPUExecutionProvider"]
IMAGE_EXT = (".png", ".jpg", ".jpeg", ".webp", ".bmp", ".jfif")


# ---- the face you become
def load_persona(path, sw):
    """(name, face) for the biggest face in the image at path, or None (no file, or no face in it)."""
    if path == PERSONA and not os.path.exists(PERSONA) and os.path.exists(PLACEHOLDER):
        path = PLACEHOLDER
    img = cv2.imread(path)
    if img is None:
        print(f"Persona: no picture at {path}: put her face there and press R")
        return None
    faces = sw.faces(img, size=640, identify=True)
    if not faces:
        # a face that fills the whole picture (a tight close-up) isn't found: give it room and look again
        h, w = img.shape[:2]
        pad = int(max(h, w) * 0.4)
        faces = sw.faces(cv2.copyMakeBorder(img, pad, pad, pad, pad, cv2.BORDER_REPLICATE), size=640, identify=True)
    if not faces:
        print(f"Persona: no face found in {path}: a clearer, front-facing photo works best")
        return None
    face = max(faces, key=lambda x: (x.bbox[2] - x.bbox[0]) * (x.bbox[3] - x.bbox[1]))
    name = os.path.splitext(os.path.basename(path))[0]
    print(f"Persona: {name} ready")
    return name, face


# ---- the camera
def open_source(args):
    """(read() -> frame or None, is_still)"""
    if args.source:
        if args.source.lower().endswith(IMAGE_EXT):
            img = cv2.imread(args.source)
            if img is None:
                sys.exit(f"Can't read {args.source}")
            return (lambda: img.copy()), True
        cap = cv2.VideoCapture(args.source)
    else:
        cap = cv2.VideoCapture(args.camera, cv2.CAP_DSHOW)
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, args.width)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, args.height)
    if not cap.isOpened():
        sys.exit("No camera: is another program using it? (--camera N picks another one)")

    def read():
        ok, frame = cap.read()
        if not ok and args.source:                  # a video: from the start again
            cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
            ok, frame = cap.read()
        return frame if ok else None
    return read, False


def open_virtual(w, h, fps):
    try:
        import pyvirtualcam
        cam = pyvirtualcam.Camera(width=w, height=h, fps=fps, fmt=pyvirtualcam.PixelFormat.BGR)
        print(f"Virtual camera: {cam.device} ({w}x{h}): pick it in your call")
        return cam
    except Exception as e:
        print(f"No virtual camera ({e}). Is OBS Studio installed? The preview still works.")
        return None


def caption(frame, lines):
    y = 24
    for text in lines:
        cv2.putText(frame, text, (12, y), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 0, 0), 3, cv2.LINE_AA)
        cv2.putText(frame, text, (12, y), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 1, cv2.LINE_AA)
        y += 22


class Blinks:
    """Your own blinks: the swap model only half-closes her eyes, so as yours close, the picture fades to
    your own closed eyelids (a closed eye shows next to nothing of who you are). MediaPipe watches the
    lids; a blink is measured against your own eyes' usual "open" (learned as you go), because eyes
    differ and MediaPipe never sees a lid quite shut: from 85 % of it closing, at 68 % shut."""
    EYES = ([33, 246, 161, 160, 159, 158, 157, 173, 133, 155, 154, 153, 145, 144, 163, 7],
            [362, 398, 384, 385, 386, 387, 388, 466, 263, 249, 390, 373, 374, 380, 381, 382])
    OPENNESS = ((159, 145, 33, 133), (386, 374, 362, 263))   # top, bottom, corner, corner
    CLOSING, SHUT = 0.85, 0.68

    def __init__(self):
        self.mesh = mp.solutions.face_mesh.FaceMesh(static_image_mode=False, max_num_faces=1,
                                                     refine_landmarks=False, min_detection_confidence=0.5)
        self.open = [None, None]

    def own(self, frame, box):
        """How much of your own eyes shows (0..1) over box (x0, y0, x1, y1), or None (no blink)."""
        x0, y0, x1, y1 = box
        crop = frame[y0:y1, x0:x1]
        res = self.mesh.process(cv2.cvtColor(crop, cv2.COLOR_BGR2RGB))
        if not res.multi_face_landmarks:
            return None
        h, w = crop.shape[:2]
        lm = np.array([(p.x * w, p.y * h) for p in res.multi_face_landmarks[0].landmark[:468]], np.float32)
        mask = None
        for k, (poly, (t, b, c1, c2)) in enumerate(zip(self.EYES, self.OPENNESS)):
            gap = np.linalg.norm(lm[t] - lm[b]) / max(np.linalg.norm(lm[c1] - lm[c2]), 1e-3)
            ref = gap if self.open[k] is None else max(self.open[k] * 0.998, gap)
            self.open[k] = ref
            shut = float(np.clip((self.CLOSING * ref - gap) / ((self.CLOSING - self.SHUT) * ref), 0, 1))
            if shut <= 0:
                continue
            pts = lm[poly]
            centre = pts.mean(0)
            pts = centre + (pts - centre) * [1.25, 1.9]       # the lid and its crease, not just the slit
            if mask is None:
                mask = np.zeros((h, w), np.float32)
            cv2.fillPoly(mask, [np.int32(np.round(pts))], shut)
        if mask is None:
            return None
        k = max(3, int(min(h, w) * 0.03) | 1)
        return cv2.GaussianBlur(mask, (k, k), 0)


class Swapper:
    """Finds the face in each frame and swaps it, steadying the face's points between frames."""

    def __init__(self):
        if not os.path.exists(SWAP_MODEL):
            sys.exit(f"The swap model isn't there yet: {SWAP_MODEL} (see README.md)")
        for f in ("det_10g.onnx", "w600k_r50.onnx"):
            if not os.path.exists(os.path.join(BUFFALO, f)):
                sys.exit(f"InsightFace's buffalo_l models aren't in {BUFFALO} (see README.md)")
        # the face finder, every frame (small and fast); the recogniser only reads the persona's identity
        print("Loading: face finder ...", flush=True)
        self.detector = insightface.model_zoo.get_model(os.path.join(BUFFALO, "det_10g.onnx"), providers=PROVIDERS)
        self.detector.prepare(ctx_id=0, input_size=(320, 320), det_thresh=0.5)
        print("Loading: face recogniser ...", flush=True)
        self.recogniser = insightface.model_zoo.get_model(os.path.join(BUFFALO, "w600k_r50.onnx"), providers=PROVIDERS)
        self.recogniser.prepare(ctx_id=0)
        print("Loading: the swap model ...", flush=True)
        self.model = insightface.model_zoo.get_model(SWAP_MODEL, providers=PROVIDERS)
        self.occluder = None
        if os.path.exists(OCCLUDER_MODEL):
            import onnxruntime as ort
            print("Loading: the occlusion model ...", flush=True)
            self.occluder = ort.InferenceSession(OCCLUDER_MODEL, providers=PROVIDERS)
            self.occluder_in = self.occluder.get_inputs()[0].name
        else:
            print("No occlusion model (models/xseg_1.onnx): hands in front of the face get painted over")
        self.occlusion_on = self.occluder is not None
        self._visible, self._frames = None, 0     # the occluder looks every other frame; between, its mask stays
        self.blinks = Blinks()
        self.kps = None              # the steadied face points of the last frame
        self.smooth = 0.5

    def faces(self, img, size=320, identify=False):
        """Every face in img (InsightFace Face objects: bbox, kps, and with identify, its identity)."""
        bboxes, kpss = self.detector.detect(img, input_size=(size, size), max_num=0, metric="default")
        out = []
        for i in range(bboxes.shape[0]):
            face = Face(bbox=bboxes[i, :4], kps=kpss[i] if kpss is not None else None, det_score=bboxes[i, 4])
            if identify:
                self.recogniser.get(img, face)
            out.append(face)
        return out

    def biggest_face(self, frame):
        faces = self.faces(frame)
        if not faces:
            self.kps = None
            return None
        face = max(faces, key=lambda x: (x.bbox[2] - x.bbox[0]) * (x.bbox[3] - x.bbox[1]))
        if self.kps is not None and self.smooth > 0:   # steadier: part last frame's points, part this one's
            moved = np.abs(face.kps - self.kps).max()
            k = self.smooth if moved < 12 else self.smooth * 0.3   # a quick move: follow it, don't drag behind
            face.kps = (self.kps * k + face.kps * (1 - k)).astype(np.float32)
        self.kps = face.kps.copy()
        return face

    def swap(self, frame, persona_face, strength=1.0):
        face = self.biggest_face(frame)
        if face is None:
            return frame, False
        fake, m = self.model.get(frame, face, persona_face, paste_back=False)
        visible = self.visible(frame, m) if self.occlusion_on else None
        out = self.paste(frame, fake, m, strength, visible)
        # your blinks: around the face, as your eyes close the picture fades to your own closed lids
        h, w = frame.shape[:2]
        bx0, by0, bx1, by1 = face.bbox.astype(int)
        pad = int((bx1 - bx0) * 0.25)
        box = (max(bx0 - pad, 0), max(by0 - pad, 0), min(bx1 + pad, w), min(by1 + pad, h))
        if box[2] - box[0] > 16 and box[3] - box[1] > 16:
            own = self.blinks.own(frame, box)
            if own is not None:
                x0, y0, x1, y1 = box
                a = own[..., None]
                out[y0:y1, x0:x1] = (frame[y0:y1, x0:x1] * a + out[y0:y1, x0:x1] * (1 - a)).astype(np.uint8)
        return out, True

    def visible(self, frame, m):
        """Where the face is really face (1) and where something's in front of it (0), on the swap's own
        128 x 128 crop. The occluder sees the same crop at 256 x 256 (m scaled up)."""
        self._frames += 1
        if self._visible is not None and self._frames % 2 == 0:
            return self._visible               # it's in the face's own frame, so it still follows the face
        size = 256
        crop = cv2.warpAffine(frame, m * (size / 128), (size, size), borderValue=0.0)
        out = self.occluder.run(None, {self.occluder_in: crop[None].astype(np.float32) / 255.0})[0][0, ..., 0]
        mask = cv2.GaussianBlur(np.clip(out, 0, 1), (0, 0), 5)
        mask = (np.clip(mask, 0.5, 1) - 0.5) * 2             # a firm edge, softened: as FaceFusion does
        self._visible = cv2.resize(mask, (128, 128), interpolation=cv2.INTER_AREA)
        return self._visible

    def paste(self, frame, fake, m, strength=1.0, visible=None):
        """The swapped face (128 x 128, the model's own crop) put back into frame, feathered at its edge.
        InsightFace's own paste_back works on the whole frame on the CPU; this only touches the face's box,
        which is most of the speed."""
        h, w = frame.shape[:2]
        im = cv2.invertAffineTransform(m)
        size = fake.shape[0]
        corners = cv2.transform(np.float32([[[0, 0], [size, 0], [0, size], [size, size]]]), im)[0]
        x0, y0 = np.floor(corners.min(0)).astype(int) - 2
        x1, y1 = np.ceil(corners.max(0)).astype(int) + 2
        x0, y0, x1, y1 = max(x0, 0), max(y0, 0), min(x1, w), min(y1, h)
        if x1 - x0 < 8 or y1 - y0 < 8:
            return frame
        im[:, 2] -= (x0, y0)                                    # the box's own coordinates
        bw, bh = x1 - x0, y1 - y0
        face = cv2.warpAffine(fake, im, (bw, bh), borderValue=0.0)
        white = np.full((size, size), 255, np.uint8)
        mask = cv2.warpAffine(white, im, (bw, bh), borderValue=0)
        mask[mask > 20] = 255
        # feather the edge in, as InsightFace does (sized by the face, so it looks the same near or far)
        ys, xs = np.nonzero(mask == 255)
        side = int(np.sqrt((ys.max() - ys.min()) * (xs.max() - xs.min()))) if len(ys) else 0
        k = max(side // 10, 10)
        # (outside the box counts as empty: the crop's corners touch the box's edge, and must shrink too)
        mask = cv2.erode(mask, np.ones((k, k), np.uint8), borderType=cv2.BORDER_CONSTANT, borderValue=0)
        blur = max(side // 20, 5)
        mask = cv2.GaussianBlur(mask, (2 * blur + 1, 2 * blur + 1), 0)
        alpha = mask.astype(np.float32) / 255.0 * strength
        if visible is not None:                                 # hands, hair... in front: her face not over them
            alpha *= cv2.warpAffine(visible, im, (bw, bh), borderValue=0.0)
        alpha = alpha[..., None]
        out = frame.copy()
        roi = out[y0:y1, x0:x1]
        out[y0:y1, x0:x1] = (face.astype(np.float32) * alpha + roi.astype(np.float32) * (1 - alpha)).astype(np.uint8)
        return out


def main():
    ap = argparse.ArgumentParser(description="Persona Cam")
    ap.add_argument("--camera", type=int, default=0)
    ap.add_argument("--source", help="a video or picture instead of the webcam")
    ap.add_argument("--persona", default=PERSONA, help="the face you become (default personas/input.jpeg)")
    ap.add_argument("--width", type=int, default=1280)
    ap.add_argument("--height", type=int, default=720)
    ap.add_argument("--fps", type=int, default=30)
    ap.add_argument("--no-virtual", action="store_true")
    ap.add_argument("--snapshot", help="save one swapped frame to this file and quit (for testing)")
    args = ap.parse_args()

    read, still = open_source(args)
    sw = Swapper()
    persona = load_persona(args.persona, sw)

    if args.snapshot:
        frame = read()
        if not persona:
            sys.exit("No persona to swap to")
        sw.smooth = 0
        out, found = sw.swap(frame, persona[1])
        cv2.imwrite(args.snapshot, out)
        print(f"Saved {args.snapshot} (face found: {found}, persona: {persona[0]})")
        return

    print("Opening the preview window and the camera ...", flush=True)
    cv2.namedWindow(WINDOW, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(WINDOW, 960, 540)
    cv2.createTrackbar("smooth %", WINDOW, 50, 90, lambda v: None)
    cv2.createTrackbar("strength %", WINDOW, 100, 100, lambda v: None)

    virtual, want_virtual = None, not args.no_virtual
    swap_on = True
    fps, t_last = 0.0, time.monotonic()
    while True:
        frame = read()
        if frame is None:
            print("The camera stopped giving pictures.")
            break
        if virtual is None and want_virtual:
            virtual = open_virtual(frame.shape[1], frame.shape[0], args.fps)
            want_virtual = virtual is not None
        sw.smooth = cv2.getTrackbarPos("smooth %", WINDOW) / 100
        strength = cv2.getTrackbarPos("strength %", WINDOW) / 100
        found = False
        out = frame
        if swap_on and persona:
            out, found = sw.swap(frame, persona[1], strength)
        if virtual is not None:
            virtual.send(out)
        now = time.monotonic()
        fps = fps * 0.9 + 0.1 / max(now - t_last, 1e-3)
        t_last = now
        view = out.copy()
        name = persona[0] if persona else "none"
        status = [f"{fps:4.1f} fps   persona: {name}   "
                  f"swap: {'on' if swap_on else 'OFF'}   occlusion: {'on' if sw.occlusion_on else 'off'}   virtual camera: {'on' if virtual is not None else 'off'}"]
        if not persona:
            status.append("No persona: put her face in personas/input.jpeg and press R")
        elif swap_on and not found:
            status.append("No face in view")
        caption(view, status)
        cv2.imshow(WINDOW, view)
        k = cv2.waitKey(1) & 0xFF
        if k in (ord("q"), 27) or cv2.getWindowProperty(WINDOW, cv2.WND_PROP_VISIBLE) < 1:
            break
        if k == ord("s"):
            swap_on = not swap_on
        elif k == ord("o") and sw.occluder is not None:
            sw.occlusion_on = not sw.occlusion_on
        elif k == ord("v"):
            if virtual is not None:
                virtual.close()
                virtual, want_virtual = None, False
            else:
                want_virtual = True
        elif k == ord("r"):
            persona = load_persona(args.persona, sw)
    if virtual is not None:
        virtual.close()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
