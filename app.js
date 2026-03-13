import vision from 'https://cdn.jsdelivr.net/npm/@mediapipe/tasks-vision@0.10.14/+esm';
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
  blinkClosed: false,
  blinkLastAt: 0,
  toneIndex: 0,
  blinkScore: 0,
  recorder: null,
};

const toneTypes = ['sine', 'triangle', 'sawtooth', 'square'];

const audio = {
  ctx: null,
  osc: null,
  filter: null,
  gain: null,
  dest: null,
  proc: null,
  recording: false,
  pendingStart: false,
  pcmL: [],
  sampleRate: 48000,
  start() {
    if (this.ctx) return;
    this.ctx = new (window.AudioContext || window.webkitAudioContext)();
    this.sampleRate = this.ctx.sampleRate;
    this.osc = this.ctx.createOscillator();
    this.filter = this.ctx.createBiquadFilter();
    this.gain = this.ctx.createGain();
    this.dest = this.ctx.createMediaStreamDestination();
    this.proc = this.ctx.createScriptProcessor(2048, 1, 1);

    this.osc.type = toneTypes[state.toneIndex];
    this.osc.frequency.value = 330;
    this.filter.type = 'lowpass';
    this.filter.frequency.value = 1800;
    this.gain.gain.value = 0.0001;

    this.osc.connect(this.filter);
    this.filter.connect(this.gain);
    this.gain.connect(this.ctx.destination);
    this.gain.connect(this.dest);
    this.dest.stream.connect = () => {};
    this.gain.connect(this.proc);
    this.proc.connect(this.ctx.destination);
    this.proc.onaudioprocess = (e) => {
      if (!this.recording) return;
      const input = e.inputBuffer.getChannelData(0);
      this.pcmL.push(new Float32Array(input));
    };

    this.osc.start();
  },
  update(pitchNorm, volNorm) {
    if (!this.ctx) return;
    const hz = 220 + pitchNorm * 660;
    const gain = 0.01 + volNorm * 0.18;
    const cutoff = 500 + volNorm * 4200;
    this.osc.frequency.setTargetAtTime(hz, this.ctx.currentTime, 0.03);
    this.gain.gain.setTargetAtTime(gain, this.ctx.currentTime, 0.04);
    this.filter.frequency.setTargetAtTime(cutoff, this.ctx.currentTime, 0.05);
  },
  cycleTone() {
    state.toneIndex = (state.toneIndex + 1) % toneTypes.length;
    if (this.osc) this.osc.type = toneTypes[state.toneIndex];
    els.toneVal.textContent = toneTypes[state.toneIndex];
  },
  async armRecording() {
    this.start();
    if (this.pendingStart || this.recording) return;
    this.pendingStart = true;
    els.recordBtn.textContent = 'REC in 0.8s';
    els.recordBtn.classList.add('recording');
    setTimeout(() => {
      if (!this.pendingStart) return;
      this.pendingStart = false;
      this.recording = true;
      this.pcmL = [];
      els.recordBtn.textContent = 'Stop';
      els.msg.textContent = 'Recording…';
    }, 800);
  },
  stopRecording() {
    if (this.pendingStart) {
      this.pendingStart = false;
      els.recordBtn.textContent = 'Record';
      els.recordBtn.classList.remove('recording');
      els.msg.textContent = 'Recording cancelled.';
      return;
    }
    if (!this.recording) return;
    this.recording = false;
    els.recordBtn.textContent = 'Record';
    els.recordBtn.classList.remove('recording');
    const wav = encodeWav(this.pcmL, this.sampleRate);
    const blob = new Blob([wav], { type: 'audio/wav' });
    const url = URL.createObjectURL(blob);
    els.player.src = url;
    els.player.hidden = false;
    els.downloadLink.href = url;
    els.downloadLink.download = `glance-note-${Date.now()}.wav`;
    els.downloadLink.textContent = 'Download WAV';
    els.downloadLink.hidden = false;
    els.msg.textContent = 'Recording ready.';
    els.player.insertAdjacentElement('afterend', els.downloadLink);
  }
};

function encodeWav(chunks, sampleRate) {
  let length = 0;
  for (const c of chunks) length += c.length;
  const pcm = new Int16Array(length);
  let offset = 0;
  for (const c of chunks) {
    for (let i = 0; i < c.length; i++) {
      const s = Math.max(-1, Math.min(1, c[i]));
      pcm[offset++] = s < 0 ? s * 0x8000 : s * 0x7fff;
    }
  }
  const buffer = new ArrayBuffer(44 + pcm.length * 2);
  const view = new DataView(buffer);
  writeStr(view, 0, 'RIFF');
  view.setUint32(4, 36 + pcm.length * 2, true);
  writeStr(view, 8, 'WAVE');
  writeStr(view, 12, 'fmt ');
  view.setUint32(16, 16, true);
  view.setUint16(20, 1, true);
  view.setUint16(22, 1, true);
  view.setUint32(24, sampleRate, true);
  view.setUint32(28, sampleRate * 2, true);
  view.setUint16(32, 2, true);
  view.setUint16(34, 16, true);
  writeStr(view, 36, 'data');
  view.setUint32(40, pcm.length * 2, true);
  let p = 44;
  for (let i = 0; i < pcm.length; i++, p += 2) view.setInt16(p, pcm[i], true);
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

  const x = clamp((lxn + rxn) * 0.5, 0, 1);
  const y = clamp((lyn + ryn) * 0.5, 0, 1);
  return { x, y };
}

function blinkScoreFromResult(result, lms) {
  let score = 0;
  const shapes = result.faceBlendshapes?.[0]?.categories || [];
  if (shapes.length) {
    const left = shapes.find(c => c.categoryName === 'eyeBlinkLeft')?.score ?? 0;
    const right = shapes.find(c => c.categoryName === 'eyeBlinkRight')?.score ?? 0;
    score = (left + right) * 0.5;
  }
  if (score > 0) return score;

  const earL = eyeAspectRatio(lms[33], lms[160], lms[158], lms[133], lms[153], lms[144]);
  const earR = eyeAspectRatio(lms[362], lms[385], lms[387], lms[263], lms[373], lms[380]);
  const ear = (earL + earR) * 0.5;
  return clamp(1 - (ear / 0.32), 0, 1);
}
function eyeAspectRatio(p1,p2,p3,p4,p5,p6){
  const d1 = dist(p2,p6), d2 = dist(p3,p5), d3 = dist(p1,p4);
  return (d1+d2)/(2*Math.max(d3,0.00001));
}
function dist(a,b){const dx=a.x-b.x, dy=a.y-b.y; return Math.hypot(dx,dy);}
function clamp(v,min,max){return Math.max(min,Math.min(max,v));}
function smooth(cur, next, amt){ return cur + (next-cur)*amt; }

async function initVision() {
  const fileset = await FilesetResolver.forVisionTasks('https://cdn.jsdelivr.net/npm/@mediapipe/tasks-vision@0.10.14/wasm');
  state.landmarker = await FaceLandmarker.createFromOptions(fileset, {
    baseOptions: {
      modelAssetPath: 'https://storage.googleapis.com/mediapipe-models/face_landmarker/face_landmarker/float16/latest/face_landmarker.task'
    },
    runningMode: 'VIDEO',
    numFaces: 1,
    outputFaceBlendshapes: true,
    minFaceDetectionConfidence: 0.5,
    minFacePresenceConfidence: 0.5,
    minTrackingConfidence: 0.5,
  });
}

async function start() {
  try {
    els.startBtn.disabled = true;
    els.msg.textContent = 'Opening camera…';
    audio.start();
    if (audio.ctx.state === 'suspended') await audio.ctx.resume();
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
      const dx = clamp((state.x - state.centerX) * 3.2 + 0.5, 0, 1);
      const rawDy = (state.y - state.centerY);
      const dy = clamp(0.5 + Math.sign(rawDy) * Math.pow(Math.abs(rawDy) * 4.0, 0.86), 0, 1);
      audio.update(dx, 1 - dy);
      updateUi(dx, 1 - dy);

      const b = blinkScoreFromResult(result, lms);
      state.blinkScore = smooth(state.blinkScore, b, 0.5);
      handleBlink(state.blinkScore);
    } else {
      els.msg.textContent = 'Face not found.';
      audio.update(0.5, 0.02);
    }
  }
  state.rafId = requestAnimationFrame(renderLoop);
}

function handleBlink(score) {
  els.blinkVal.textContent = score.toFixed(2);
  const now = performance.now();
  const closed = score > 0.52;
  if (closed && !state.blinkClosed) {
    state.blinkClosed = true;
  }
  if (!closed && state.blinkClosed) {
    state.blinkClosed = false;
    const gap = now - state.blinkLastAt;
    if (gap > 110 && gap < 700) {
      audio.cycleTone();
      els.msg.textContent = `Tone: ${toneTypes[state.toneIndex]}`;
      state.blinkLastAt = 0;
      return;
    }
    state.blinkLastAt = now;
  }
}

function updateUi(pitchNorm, volNorm) {
  els.pitchVal.textContent = Math.round(220 + pitchNorm * 660) + 'Hz';
  els.volumeVal.textContent = volNorm.toFixed(2);
  const x = (pitchNorm - 0.5) * 110;
  const y = ((1 - volNorm) - 0.5) * 110;
  els.dot.style.transform = `translate(calc(-50% + ${x}px), calc(-50% + ${y}px))`;
  els.toneVal.textContent = toneTypes[state.toneIndex];
}

els.startBtn.addEventListener('click', start);
els.calBtn.addEventListener('click', () => { els.msg.textContent = 'Look at center…'; calibrate(1500); });
els.recordBtn.addEventListener('click', () => {
  if (audio.recording || audio.pendingStart) audio.stopRecording();
  else audio.armRecording();
});

if ('serviceWorker' in navigator) {
  navigator.serviceWorker.register('./sw.js?v=20260313r1').then(async reg => {
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
