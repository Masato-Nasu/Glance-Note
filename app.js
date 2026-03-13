
import * as vision from 'https://cdn.jsdelivr.net/npm/@mediapipe/tasks-vision@0.10.14/+esm';
const { FaceLandmarker, FilesetResolver } = vision;

const els = {
  video: document.getElementById('video'),
  startBtn: document.getElementById('startBtn'),
  calBtn: document.getElementById('calBtn'),
  recordBtn: document.getElementById('recordBtn'),
  msg: document.getElementById('msg'),
  dot: document.getElementById('dot'),
  swState: document.getElementById('swState'),
  modeState: document.getElementById('modeState'),
  pitchVal: document.getElementById('pitchVal'),
  volumeVal: document.getElementById('volumeVal'),
  blinkVal: document.getElementById('blinkVal'),
  toneVal: document.getElementById('toneVal'),
  player: document.getElementById('player'),
  downloadLink: document.getElementById('downloadLink'),
};

const state = {
  running: false,
  calibrating: false,
  landmarker: null,
  stream: null,
  rafId: 0,
  lastVideoTime: -1,
  centerX: 0.5,
  centerY: 0.5,
  x: 0.5,
  y: 0.5,
};

const audio = {
  ctx: null,
  osc: null,
  filter: null,
  gain: null,
  dest: null,
  rec: null,
  recChunks: [],
  recording: false,
  pendingStart: false,
  start() {
    if (this.ctx) return;
    this.ctx = new (window.AudioContext || window.webkitAudioContext)();
    this.osc = this.ctx.createOscillator();
    this.filter = this.ctx.createBiquadFilter();
    this.gain = this.ctx.createGain();
    this.dest = this.ctx.createMediaStreamDestination();

    this.osc.type = 'sine';
    this.osc.frequency.value = 330;
    this.filter.type = 'lowpass';
    this.filter.frequency.value = 1800;
    this.gain.gain.value = 0.0001;

    this.osc.connect(this.filter);
    this.filter.connect(this.gain);
    this.gain.connect(this.ctx.destination);
    this.gain.connect(this.dest);
    this.osc.start();
  },
  async unlockForIOS() {
    this.start();
    if (!this.ctx) return;
    try {
      if (this.ctx.state !== 'running') await this.ctx.resume();
    } catch (e) {
      console.warn('resume failed', e);
    }
    const primeGain = this.ctx.createGain();
    primeGain.gain.value = 0.00001;
    const primeOsc = this.ctx.createOscillator();
    primeOsc.frequency.value = 440;
    primeOsc.connect(primeGain);
    primeGain.connect(this.ctx.destination);
    const now = this.ctx.currentTime;
    primeOsc.start(now);
    primeOsc.stop(now + 0.02);
    const buffer = this.ctx.createBuffer(1, 1, 22050);
    const source = this.ctx.createBufferSource();
    source.buffer = buffer;
    source.connect(primeGain);
    source.start(now);
    try {
      await new Promise(resolve => setTimeout(resolve, 0));
    } catch {}
  },
  update(pitchNorm, volNorm) {
    if (!this.ctx) return;
    const hz = 220 + pitchNorm * 660;
    const gain = 0.005 + volNorm * 0.2;
    const cutoff = 800 + volNorm * 3200;
    this.osc.frequency.setTargetAtTime(hz, this.ctx.currentTime, 0.025);
    this.gain.gain.setTargetAtTime(gain, this.ctx.currentTime, 0.035);
    this.filter.frequency.setTargetAtTime(cutoff, this.ctx.currentTime, 0.045);
  },
  async armRecording() {
    this.start();
    if (this.pendingStart || this.recording) return;
    this.pendingStart = true;
    els.recordBtn.textContent = 'REC in 0.8s';
    els.recordBtn.classList.add('recording');
    setTimeout(async () => {
      if (!this.pendingStart) return;
      this.pendingStart = false;
      try {
        await this.beginMediaRecorder();
        this.recording = true;
        els.recordBtn.textContent = 'Stop';
        els.msg.textContent = 'Recording…';
      } catch (e) {
        console.error(e);
        this.recording = false;
        els.recordBtn.textContent = 'Record';
        els.recordBtn.classList.remove('recording');
        els.msg.textContent = 'Could not start recording.';
      }
    }, 800);
  },
  async beginMediaRecorder() {
    this.recChunks = [];
    let mimeType = '';
    const candidates = [
      'audio/webm;codecs=opus',
      'audio/webm',
      'audio/ogg;codecs=opus',
      'audio/mp4'
    ];
    for (const c of candidates) {
      if (window.MediaRecorder && MediaRecorder.isTypeSupported(c)) {
        mimeType = c;
        break;
      }
    }
    this.rec = mimeType ? new MediaRecorder(this.dest.stream, { mimeType }) : new MediaRecorder(this.dest.stream);
    this.rec.ondataavailable = (e) => {
      if (e.data && e.data.size > 0) this.recChunks.push(e.data);
    };
    this.rec.start(250);
  },
  async stopRecording() {
    if (this.pendingStart) {
      this.pendingStart = false;
      els.recordBtn.textContent = 'Record';
      els.recordBtn.classList.remove('recording');
      els.msg.textContent = 'Recording cancelled.';
      return;
    }
    if (!this.recording || !this.rec) return;
    const rec = this.rec;
    const chunks = this.recChunks;
    this.recording = false;
    this.rec = null;
    els.recordBtn.textContent = 'Record';
    els.recordBtn.classList.remove('recording');

    const blob = await new Promise((resolve) => {
      rec.onstop = () => resolve(new Blob(chunks, { type: rec.mimeType || 'audio/webm' }));
      rec.stop();
    });

    const wavBlob = await convertBlobToWav(blob);
    const url = URL.createObjectURL(wavBlob);
    els.player.src = url;
    els.player.hidden = false;
    els.downloadLink.href = url;
    els.downloadLink.download = `glance-note-${Date.now()}.wav`;
    els.downloadLink.textContent = 'Download WAV';
    els.downloadLink.hidden = false;
    els.player.insertAdjacentElement('afterend', els.downloadLink);
    els.msg.textContent = 'Recording ready.';
  }
};

async function convertBlobToWav(blob) {
  const arrayBuffer = await blob.arrayBuffer();
  const decodeCtx = new (window.AudioContext || window.webkitAudioContext)();
  try {
    const audioBuffer = await decodeCtx.decodeAudioData(arrayBuffer.slice(0));
    const wav = encodeWavFromAudioBuffer(audioBuffer);
    return new Blob([wav], { type: 'audio/wav' });
  } finally {
    if (decodeCtx.close) decodeCtx.close();
  }
}

function encodeWavFromAudioBuffer(audioBuffer) {
  const channels = Math.min(audioBuffer.numberOfChannels, 2);
  const sampleRate = audioBuffer.sampleRate;
  const length = audioBuffer.length;
  const interleaved = new Int16Array(length * channels);
  if (channels === 1) {
    const ch0 = audioBuffer.getChannelData(0);
    for (let i = 0; i < length; i++) {
      const s = Math.max(-1, Math.min(1, ch0[i]));
      interleaved[i] = s < 0 ? s * 0x8000 : s * 0x7fff;
    }
  } else {
    const ch0 = audioBuffer.getChannelData(0);
    const ch1 = audioBuffer.getChannelData(1);
    let o = 0;
    for (let i = 0; i < length; i++) {
      let s0 = Math.max(-1, Math.min(1, ch0[i]));
      let s1 = Math.max(-1, Math.min(1, ch1[i]));
      interleaved[o++] = s0 < 0 ? s0 * 0x8000 : s0 * 0x7fff;
      interleaved[o++] = s1 < 0 ? s1 * 0x8000 : s1 * 0x7fff;
    }
  }
  const buffer = new ArrayBuffer(44 + interleaved.length * 2);
  const view = new DataView(buffer);
  writeStr(view, 0, 'RIFF');
  view.setUint32(4, 36 + interleaved.length * 2, true);
  writeStr(view, 8, 'WAVE');
  writeStr(view, 12, 'fmt ');
  view.setUint32(16, 16, true);
  view.setUint16(20, 1, true);
  view.setUint16(22, channels, true);
  view.setUint32(24, sampleRate, true);
  view.setUint32(28, sampleRate * channels * 2, true);
  view.setUint16(32, channels * 2, true);
  view.setUint16(34, 16, true);
  writeStr(view, 36, 'data');
  view.setUint32(40, interleaved.length * 2, true);
  let p = 44;
  for (let i = 0; i < interleaved.length; i++, p += 2) view.setInt16(p, interleaved[i], true);
  return buffer;
}
function writeStr(view, offset, s) { for (let i = 0; i < s.length; i++) view.setUint8(offset + i, s.charCodeAt(i)); }

function avgLandmarks(lms, ids) {
  let x = 0, y = 0;
  for (const id of ids) { x += lms[id].x; y += lms[id].y; }
  return { x: x / ids.length, y: y / ids.length };
}

function irisControl(lms) {
  const L_IRIS = [468,469,470,471,472];
  const R_IRIS = [473,474,475,476,477];
  const li = avgLandmarks(lms, L_IRIS);
  const ri = avgLandmarks(lms, R_IRIS);

  const lOuter = lms[33], lInner = lms[133], lTop = lms[159], lBot = lms[145];
  const rInner = lms[362], rOuter = lms[263], rTop = lms[386], rBot = lms[374];

  const lxn = (li.x - lOuter.x) / Math.max(0.0001, (lInner.x - lOuter.x));
  const rxn = (ri.x - rInner.x) / Math.max(0.0001, (rOuter.x - rInner.x));
  const lyn = (li.y - lTop.y) / Math.max(0.0001, (lBot.y - lTop.y));
  const ryn = (ri.y - rTop.y) / Math.max(0.0001, (rBot.y - rTop.y));

  return {
    x: clamp((lxn + rxn) * 0.5, 0, 1),
    y: clamp((lyn + ryn) * 0.5, 0, 1)
  };
}

function clamp(v, min, max) { return Math.max(min, Math.min(max, v)); }
function smooth(cur, next, amt) { return cur + (next - cur) * amt; }

async function initVision() {
  const fileset = await FilesetResolver.forVisionTasks('https://cdn.jsdelivr.net/npm/@mediapipe/tasks-vision@0.10.14/wasm');
  state.landmarker = await FaceLandmarker.createFromOptions(fileset, {
    baseOptions: {
      modelAssetPath: 'https://storage.googleapis.com/mediapipe-models/face_landmarker/face_landmarker/float16/latest/face_landmarker.task'
    },
    runningMode: 'VIDEO',
    numFaces: 1,
    outputFaceBlendshapes: false,
    minFaceDetectionConfidence: 0.5,
    minFacePresenceConfidence: 0.5,
    minTrackingConfidence: 0.5,
  });
}

async function start() {
  try {
    els.startBtn.disabled = true;
    els.msg.textContent = 'Opening camera…';
    await audio.unlockForIOS();
    if (audio.ctx && audio.ctx.state === 'suspended') await audio.ctx.resume();
    await initVision();
    state.stream = await navigator.mediaDevices.getUserMedia({
      video: { facingMode: 'user', width: { ideal: 960 }, height: { ideal: 720 } },
      audio: false
    });
    els.video.srcObject = state.stream;
    await els.video.play();
    state.running = true;
    els.modeState.textContent = 'Camera: on';
    els.calBtn.disabled = false;
    els.recordBtn.disabled = false;
    els.msg.textContent = 'Look at center for 2 seconds.';
    calibrate(2000);
    renderLoop();
  } catch (err) {
    console.error(err);
    els.msg.textContent = 'Could not start camera.';
    els.startBtn.disabled = false;
  }
}

function calibrate(ms = 1500) {
  state.calibrating = true;
  let sx = 0, sy = 0, n = 0;
  const started = performance.now();
  const tick = () => {
    if (!state.running) return;
    if (performance.now() - started >= ms) {
      if (n > 0) {
        state.centerX = sx / n;
        state.centerY = sy / n;
      }
      state.calibrating = false;
      els.msg.textContent = 'Ready.';
      return;
    }
    if (state._latestIris) { sx += state._latestIris.x; sy += state._latestIris.y; n++; }
    requestAnimationFrame(tick);
  };
  requestAnimationFrame(tick);
}

function renderLoop() {
  if (!state.running) return;
  const t = performance.now();
  if (els.video.currentTime !== state.lastVideoTime) {
    state.lastVideoTime = els.video.currentTime;
    const result = state.landmarker.detectForVideo(els.video, t);
    const lms = result.faceLandmarks?.[0];
    if (lms) {
      const iris = irisControl(lms);
      state._latestIris = iris;
      state.x = smooth(state.x, iris.x, 0.22);
      state.y = smooth(state.y, iris.y, 0.18);
      const pitchNorm = clamp(0.5 - (state.x - state.centerX) * 3.2, 0, 1);
      const rawDy = (state.y - state.centerY);
      const yMapped = clamp(0.5 + Math.sign(rawDy) * Math.pow(Math.abs(rawDy) * 4.0, 0.86), 0, 1);
      const volNorm = 1 - yMapped;
      audio.update(pitchNorm, volNorm);
      updateUi(pitchNorm, volNorm);
    } else {
      els.msg.textContent = 'Face not found.';
      audio.update(0.5, 0.02);
    }
  }
  state.rafId = requestAnimationFrame(renderLoop);
}

function updateUi(pitchNorm, volNorm) {
  els.pitchVal.textContent = Math.round(220 + pitchNorm * 660) + 'Hz';
  els.volumeVal.textContent = volNorm.toFixed(2);
  els.blinkVal.textContent = 'sine';
  els.toneVal.textContent = 'sine';
  const x = (pitchNorm - 0.5) * 110;
  const y = ((1 - volNorm) - 0.5) * 110;
  els.dot.style.transform = `translate(calc(-50% + ${x}px), calc(-50% + ${y}px))`;
}

els.startBtn.addEventListener('click', start);
els.calBtn.addEventListener('click', () => { els.msg.textContent = 'Look at center…'; calibrate(1500); });
els.recordBtn.addEventListener('click', () => {
  if (audio.recording || audio.pendingStart) audio.stopRecording();
  else audio.armRecording();
});


async function reviveAudio() {
  if (!audio.ctx) return;
  try {
    if (audio.ctx.state !== 'running') {
      await audio.ctx.resume();
      await audio.unlockForIOS();
    }
  } catch (e) {
    console.warn('audio revive failed', e);
  }
}

document.addEventListener('visibilitychange', () => {
  if (document.visibilityState === 'visible') reviveAudio();
});
window.addEventListener('pageshow', () => { reviveAudio(); });

if ('serviceWorker' in navigator) {
  navigator.serviceWorker.register('./sw.js?v=20260313r4').then(async reg => {
    els.swState.textContent = 'SW: registered';
    await navigator.serviceWorker.ready;
    els.swState.textContent = 'SW: ready';
  }).catch(err => {
    console.error(err);
    els.swState.textContent = 'SW: error';
  });
} else {
  els.swState.textContent = 'SW: n/a';
}
