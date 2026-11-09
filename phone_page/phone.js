// The phone's side of Changeling Imago Realtime: the camera goes to the PC a picture at a time (each answer is
// the swapped picture, shown big), and the microphone in short pieces of 16-bit sound at 24 kHz.
(function () {
  "use strict";
  var RATE = 24000;               // what the PC's voice changer works at
  var WIDTH = 640;                // pictures are sent at most this wide
  var video = document.getElementById("camera");
  var swapped = document.getElementById("swapped");
  var hint = document.getElementById("hint");
  var status = document.getElementById("status");
  var startBtn = document.getElementById("start");
  var micBox = document.getElementById("mic");
  var facing = "user";
  var stream = null, running = false, sound = null;
  var canvas = document.createElement("canvas");
  var frames = 0, since = performance.now();

  function say(text) { status.textContent = text; }

  function openCamera() {
    if (stream) stream.getTracks().forEach(function (t) { t.stop(); });
    return navigator.mediaDevices.getUserMedia({
      video: { facingMode: facing, width: { ideal: 960 }, height: { ideal: 720 } },
      audio: micBox.checked ? { echoCancellation: true, noiseSuppression: true, channelCount: 1 } : false
    }).then(function (s) {
      stream = s;
      video.srcObject = s;
      video.style.display = "block";
      hint.style.display = "none";
      return video.play();
    });
  }

  // one picture out, the swapped one back, then the next: the pace follows the PC
  function loop() {
    if (!running) return;
    if (!video.videoWidth) { setTimeout(loop, 100); return; }
    var k = Math.min(1, WIDTH / video.videoWidth);
    canvas.width = Math.round(video.videoWidth * k);
    canvas.height = Math.round(video.videoHeight * k);
    var g = canvas.getContext("2d");
    if (facing === "user") { g.translate(canvas.width, 0); g.scale(-1, 1); }   // a mirror, as a webcam is
    g.drawImage(video, 0, 0, canvas.width, canvas.height);
    canvas.toBlob(function (blob) {
      fetch("frame", { method: "POST", body: blob, headers: { "Content-Type": "image/jpeg" } })
        .then(function (r) { return r.blob(); })
        .then(function (b) {
          if (b.size) {
            var old = swapped.src;
            swapped.src = URL.createObjectURL(b);
            swapped.style.display = "block";
            if (old) URL.revokeObjectURL(old);
          }
          frames += 1;
          var now = performance.now();
          if (now - since > 1000) {
            say(Math.round(frames * 1000 / (now - since)) + " pictures a second" + (sound ? ", microphone on" : ""));
            frames = 0; since = now;
          }
          loop();
        })
        .catch(function () { say("Lost the PC: is Imago Realtime still running?"); setTimeout(loop, 1000); });
    }, "image/jpeg", 0.7);
  }

  // the microphone: 16-bit pieces of about a tenth of a second
  function startSound() {
    if (!micBox.checked || !stream || !stream.getAudioTracks().length) return;
    var Ctx = window.AudioContext || window.webkitAudioContext;
    var ctx;
    try { ctx = new Ctx({ sampleRate: RATE }); } catch (e) { ctx = new Ctx(); }
    var src = ctx.createMediaStreamSource(stream);
    var node = ctx.createScriptProcessor(2048, 1, 1);
    var step = ctx.sampleRate / RATE;            // 1 when the browser gave us 24 kHz
    node.onaudioprocess = function (e) {
      var x = e.inputBuffer.getChannelData(0);
      var n = Math.floor(x.length / step);
      var pcm = new Int16Array(n);
      for (var i = 0; i < n; i++) {
        var v = x[Math.floor(i * step)];
        pcm[i] = v < -1 ? -32768 : v > 1 ? 32767 : v * 32767;
      }
      fetch("audio", { method: "POST", body: pcm.buffer, headers: { "Content-Type": "application/octet-stream" } })
        .catch(function () {});
    };
    src.connect(node);
    node.connect(ctx.destination);               // (silent: nothing is written to the output)
    sound = { ctx: ctx, node: node, src: src };
  }

  function stopSound() {
    if (!sound) return;
    sound.node.disconnect(); sound.src.disconnect(); sound.ctx.close();
    sound = null;
  }

  startBtn.addEventListener("click", function () {
    if (running) {
      running = false;
      stopSound();
      if (stream) stream.getTracks().forEach(function (t) { t.stop(); });
      startBtn.textContent = "Start";
      say("Stopped.");
      return;
    }
    say("Opening the camera ...");
    openCamera().then(function () {
      running = true;
      startBtn.textContent = "Stop";
      startSound();
      loop();
    }).catch(function (e) {
      say("The camera didn't open: " + e.message + ". Allow it for this page (and accept the certificate warning).");
    });
  });

  document.getElementById("flip").addEventListener("click", function () {
    facing = facing === "user" ? "environment" : "user";
    if (running) { stopSound(); openCamera().then(startSound); }
  });

  micBox.addEventListener("change", function () {
    if (running) { stopSound(); openCamera().then(startSound); }
  });
})();
