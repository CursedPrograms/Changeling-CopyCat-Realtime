# Persona Cam

A private project: a live face swap for video calls. You become the face in `personas/input.jpeg`, with
your own expressions (your smile, your blinks, your mouth as you talk), and your call sees it as an
ordinary camera.

## Use it

1. Put her face in `personas/input.jpeg` (see below). Without it, `personas/placeholder.jpeg` is used.
2. Start `run.bat`. The first time it sets itself up: Python 3.11, PyTorch's CUDA libraries, InsightFace
   and the GPU ONNX runtime (about 3 GB), and the swap model (`models/inswapper_128.onnx`, about 550 MB).
3. In your call, pick the camera **OBS Virtual Camera** (OBS Studio installed; it doesn't have to run).

| Key | Does |
| :--- | :--- |
| <kbd>S</kbd> | swap on / off |
| <kbd>O</kbd> | occlusion on / off: hands, hair, a cup in front of your face stay in front of hers |
| <kbd>V</kbd> | virtual camera on / off |
| <kbd>R</kbd> | read `personas/input.jpeg` again (after replacing it) |
| <kbd>Q</kbd> or <kbd>Esc</kbd> | quit |

| Slider | Does |
| :--- | :--- |
| smooth % | steadies where the face sits from frame to frame (less jitter; too much and it lags) |
| strength % | how much of the swapped face shows: 100 is all of it |

`run.bat --camera 1` (another webcam), `run.bat --source clip.mp4` (a video or picture instead of the
webcam), `run.bat --no-virtual` (just the preview), `run.bat --persona other.jpg` (become another face for this run), `run.bat --source me.jpg --snapshot out.png` (one
swapped frame, to try a persona).

## The personas

Generated people who don't exist (AI Visual Workshop, Gemini...), not photos of real people. What works best:

- looking straight at the camera, the whole face visible: no hands or hair over the eyes or mouth
- even light, sharp, a neutral face or a slight smile
- one face per image (if there are more, the biggest is used); a tight close-up is fine too

`personas/input.jpeg` is git-ignored (hers stays on this PC); `personas/placeholder.jpeg` is the repo's.

## What it swaps, and what it doesn't

- The face: the persona's features and skin, your expressions and head movements.
- Not your hair, neck, body or voice: a hood or a wig helps a lot. A voice changer would be its own step.
- Best facing the camera: sharp side profiles and hands over the face make it slip for a moment.

## How it works

InsightFace finds the face in every frame (SCRFD, from the `buffalo_l` models in `~/.insightface`), its
five key points are steadied against the last frame's, and `inswapper_128` draws the persona's identity
(taken once from her image with InsightFace's recognition model) onto your face, all on the GPU through
onnxruntime-gpu. The finished frame goes to the preview and to OBS's virtual camera.

InsightFace's models (`buffalo_l`, `inswapper_128`) are for non-commercial use only.
