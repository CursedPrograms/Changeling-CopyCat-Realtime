"""phone.py - your phone as the camera and the microphone; this PC does the work.

    run_server.bat            (= run.bat --phone)  then scan the QR code in the window with your phone

A small web server on this PC (Flask, over HTTPS: phones only let a page use the camera and microphone on a secure
connection, so it makes its own certificate the first time; the phone warns that it's self-signed once, and you
accept it). The page sends the phone's camera a picture at a time and its microphone in short pieces; the swap
runs here, as with the webcam, out to OBS Virtual Camera, the voice changer and recordings; and the swapped picture
goes back to the phone's screen.

The port is 8443 (--phone-port), outside 5000 - 5999 (those are DREAM Robotics'). Windows asks once whether
Python may take connections on your network: allow it on private networks.
"""
import datetime
import ipaddress
import os
import socket
import threading
import time

import cv2
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
PAGE = os.path.join(HERE, "phone_page")
CERTS = os.path.join(HERE, "certs")


def lan_ip():
    """This PC's address on the local network (the one a phone on the same WiFi reaches)."""
    s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    try:
        s.connect(("192.168.0.1", 9))             # nothing is sent: it only picks the interface
        return s.getsockname()[0]
    except OSError:
        return "127.0.0.1"
    finally:
        s.close()


def certificate(ip):
    """(cert, key) paths: a self-signed certificate for this PC's address, made once (again if the address changes)."""
    os.makedirs(CERTS, exist_ok=True)
    cert, key = os.path.join(CERTS, "phone.pem"), os.path.join(CERTS, "phone.key")
    stamp = os.path.join(CERTS, "for.txt")
    if os.path.exists(cert) and os.path.exists(stamp) and open(stamp).read().strip() == ip:
        return cert, key
    from cryptography import x509
    from cryptography.hazmat.primitives import hashes, serialization
    from cryptography.hazmat.primitives.asymmetric import rsa
    from cryptography.x509.oid import NameOID

    k = rsa.generate_private_key(public_exponent=65537, key_size=2048)
    name = x509.Name([x509.NameAttribute(NameOID.COMMON_NAME, "Changeling Imago Realtime")])
    now = datetime.datetime.now(datetime.timezone.utc)
    sans = [x509.DNSName("localhost"), x509.IPAddress(ipaddress.ip_address(ip))]
    c = (x509.CertificateBuilder().subject_name(name).issuer_name(name).public_key(k.public_key())
         .serial_number(x509.random_serial_number()).not_valid_before(now - datetime.timedelta(days=1))
         .not_valid_after(now + datetime.timedelta(days=825))
         .add_extension(x509.SubjectAlternativeName(sans), critical=False).sign(k, hashes.SHA256()))
    open(cert, "wb").write(c.public_bytes(serialization.Encoding.PEM))
    open(key, "wb").write(k.private_bytes(serialization.Encoding.PEM, serialization.PrivateFormat.TraditionalOpenSSL,
                                          serialization.NoEncryption()))
    open(stamp, "w").write(ip)
    return cert, key


def waiting_picture(url, size=(1280, 720)):
    """What the window shows until the phone connects: the address, big."""
    w, h = size
    img = np.full((h, w, 3), (26, 22, 22), np.uint8)
    for i, (text, scale, colour) in enumerate((("Open this on your phone:", 1.0, (200, 200, 200)),
                                               (url, 1.3, (138, 64, 224)),
                                               ("(or scan the QR code on the right)", 0.8, (150, 150, 150)))):
        (tw, th), _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, scale, 2)
        cv2.putText(img, text, ((w - tw) // 2, h // 2 - 60 + i * 70), cv2.FONT_HERSHEY_SIMPLEX, scale, colour, 2,
                    cv2.LINE_AA)
    return img


class Phone:
    """The server. read() is the newest picture from the phone (the camera source); show(frame) sends the swapped
    picture back; voice(feed) passes the phone's sound (float32 mono, 24 kHz pieces) to the voice changer."""

    def __init__(self, port=8443, rate=24000):
        self.port, self.rate = port, rate
        self.ip = lan_ip()
        self.url = f"https://{self.ip}:{port}/"
        self.frame = None
        self.size = 640                  # the longer side of the phone's pictures: the swapped ones go back as big
        self.last_seen = 0.0
        self.out_jpeg = None
        self.feed = None
        self._waiting = waiting_picture(self.url)
        self._lock = threading.Lock()

    @property
    def connected(self):
        return time.monotonic() - self.last_seen < 2.0

    def read(self):
        time.sleep(1 / 60)                      # the worker loops on this: don't spin
        with self._lock:
            frame = self.frame
        return frame.copy() if frame is not None and self.connected else self._waiting.copy()

    def show(self, frame):
        if not self.connected:
            return
        h, w = frame.shape[:2]
        k = min(self.size, 1280) / max(w, h)       # as big as the phone sent it (HD on the phone: up to 1280)
        small = cv2.resize(frame, (int(w * k), int(h * k)), interpolation=cv2.INTER_AREA) if k < 1 else frame
        ok, buf = cv2.imencode(".jpg", small, [cv2.IMWRITE_JPEG_QUALITY, 88 if self.size > 640 else 75])
        if ok:
            self.out_jpeg = buf.tobytes()

    def voice(self, feed):
        self.feed = feed

    def qr(self, size=220):
        """The address as a QR code (RGB numpy), pink on white."""
        import segno
        from PIL import Image
        path = os.path.join(CERTS, "qr.png")
        segno.make(self.url, error="m").save(path, scale=8, border=2, dark="#005184", light="#ffffff")
        return np.array(Image.open(path).convert("RGB").resize((size, size), Image.NEAREST))

    def start(self):
        from flask import Flask, Response, request, send_from_directory
        import logging
        logging.getLogger("werkzeug").setLevel(logging.ERROR)
        app = Flask(__name__)

        @app.get("/")
        def page():
            return send_from_directory(PAGE, "index.html")

        @app.get("/<path:name>")
        def files(name):
            return send_from_directory(PAGE, name)

        @app.post("/frame")
        def frame():
            img = cv2.imdecode(np.frombuffer(request.get_data(), np.uint8), cv2.IMREAD_COLOR)
            if img is not None:
                with self._lock:
                    self.frame = img
                    self.size = max(img.shape[:2])
                self.last_seen = time.monotonic()
            return Response(self.out_jpeg or b"", mimetype="image/jpeg")   # the last swapped picture back

        @app.post("/audio")
        def audio():
            pcm = np.frombuffer(request.get_data(), "<i2").astype(np.float32) / 32768.0
            if self.feed is not None and len(pcm):
                self.feed(pcm)
            return Response(status=204)

        cert, key = certificate(self.ip)
        threading.Thread(target=lambda: app.run(host="0.0.0.0", port=self.port, ssl_context=(cert, key),
                                                threaded=True, use_reloader=False),
                         daemon=True, name="phone").start()
        print(f"Phone: open {self.url} on your phone (same WiFi), accept the certificate warning once", flush=True)
        return self
