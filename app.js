import { FaceLandmarker, FilesetResolver } from 'https://cdn.jsdelivr.net/npm/@mediapipe/tasks-vision@latest/vision_bundle.mjs';

const MODEL_URL = 'https://storage.googleapis.com/mediapipe-models/face_landmarker/face_landmarker/float16/1/face_landmarker.task';
const WASM_URL = 'https://cdn.jsdelivr.net/npm/@mediapipe/tasks-vision@latest/wasm';

const EYE = {
  left: { top: 159, bottom: 145, left: 33, right: 133, iris: [468, 469, 470, 471, 472] },
  right: { top: 386, bottom: 374, left: 362, right: 263, iris: [473, 474, 475, 476, 477] },
};

const state = {
  audioReady: false,
  cameraReady: false,
  faceReady: false,
  faceLandmarker: null,
  stream: null,
  lastVideoTime: -1,
  running: false,
  normalizedX: 0.5,
  normalizedY: 0.5,
  smoothX: 0.5,
  smoothY: 0.5,
  faceBox: null,
  calibration: null,
  lastFaceSeenAt: 0,
  eyeClosed: false,
  blinkEvents: [],
  lastBlinkHandledAt: 0,
  eyeClosedAt: 0,
  earBaseline: null,
  earSmoothed: null,
  voiceIndex: 0,
  voices: ['sine', 'triangle', 'sawtooth', 'square'],
  recorderNode: null,
  recorderBuffer: [],
  recorderSilentGain: null,
  pendingRecordTimer: null,
  recordingState: 'idle',
  recordingSampleRate: 44100,
};

const els = {
  video: document.getElementById('camera'),
  stateText: document.getElementById('stateText'),
  voiceText: document.getElementById('voiceText'),
  message: document.getElementById('message'),
  recordButton: document.getElementById('recordButton'),
  recIndicator: document.getElementById('recIndicator'),
  countdownText: document.getElementById('countdownText'),
  downloadLink: document.getElementById('downloadLink'),
  pulseRing: document.getElementById('pulseRing'),
  swText: document.getElementById('swText'),
  playback: document.getElementById('playback'),
};

let audioCtx;
let masterGain;
let osc;
let filterNode;
let outputGain;
let analyser;
let animationHandle;

function setMessage(text) {
  els.message.textContent = text;
}

function pulse() {
  els.pulseRing.classList.remove('active');
  void els.pulseRing.offsetWidth;
  els.pulseRing.classList.add('active');
}

function clamp(value, min, max) {
  return Math.min(max, Math.max(min, value));
}

function dist(a, b) {
  return Math.hypot(a.x - b.x, a.y - b.y);
}

function computeEAR(landmarks, eye) {
  const top = landmarks[eye.top];
  const bottom = landmarks[eye.bottom];
  const left = landmarks[eye.left];
  const right = landmarks[eye.right];
  const vertical = dist(top, bottom);
  const horizontal = dist(left, right) || 1e-6;
  return vertical / horizontal;
}

function averagePoints(landmarks, indices) {
  let x = 0;
  let y = 0;
  let count = 0;
  for (const idx of indices) {
    const p = landmarks[idx];
    if (!p) continue;
    x += p.x;
    y += p.y;
    count += 1;
  }
  if (!count) return null;
  return { x: x / count, y: y / count };
}

function irisRelative(landmarks, eye) {
  const iris = averagePoints(landmarks, eye.iris);
  const left = landmarks[eye.left];
  const right = landmarks[eye.right];
  const top = landmarks[eye.top];
  const bottom = landmarks[eye.bottom];
  if (!iris || !left || !right || !top || !bottom) return null;

  const xDen = (right.x - left.x) || 1e-6;
  const yDen = (bottom.y - top.y) || 1e-6;

  const nx = clamp((iris.x - left.x) / xDen, 0, 1);
  const ny = clamp((iris.y - top.y) / yDen, 0, 1);
  return { x: nx, y: ny };
}

function updateVoiceLabel() {
  els.voiceText.textContent = state.voices[state.voiceIndex].replace('sawtooth', 'saw');
}

function setAppState(label) {
  els.stateText.textContent = label;
}

function setSwState(label) {
  if (els.swText) els.swText.textContent = label;
}

function setupAudio() {
  if (state.audioReady) return;
  audioCtx = new (window.AudioContext || window.webkitAudioContext)();

  osc = audioCtx.createOscillator();
  osc.type = state.voices[state.voiceIndex];
  filterNode = audioCtx.createBiquadFilter();
  filterNode.type = 'lowpass';
  filterNode.frequency.value = 1400;
  filterNode.Q.value = 0.8;

  outputGain = audioCtx.createGain();
  outputGain.gain.value = 0.0;

  masterGain = audioCtx.createGain();
  masterGain.gain.value = 0.8;

  analyser = audioCtx.createAnalyser();
  analyser.fftSize = 1024;


  osc.connect(filterNode);
  filterNode.connect(outputGain);
  outputGain.connect(masterGain);
  masterGain.connect(audioCtx.destination);
  masterGain.connect(analyser);
  osc.start();

  state.audioReady = true;
}

function interleaveChannels(channels) {
  if (!channels.length) return new Float32Array(0);
  if (channels.length === 1) return channels[0];
  const length = channels[0].length;
  const out = new Float32Array(length * channels.length);
  for (let i = 0; i < length; i += 1) {
    for (let ch = 0; ch < channels.length; ch += 1) {
      out[i * channels.length + ch] = channels[ch][i] || 0;
    }
  }
  return out;
}

function encodeWav(samples, sampleRate) {
  const bytesPerSample = 2;
  const blockAlign = bytesPerSample;
  const buffer = new ArrayBuffer(44 + samples.length * bytesPerSample);
  const view = new DataView(buffer);

  function writeString(offset, string) {
    for (let i = 0; i < string.length; i += 1) {
      view.setUint8(offset + i, string.charCodeAt(i));
    }
  }

  writeString(0, 'RIFF');
  view.setUint32(4, 36 + samples.length * bytesPerSample, true);
  writeString(8, 'WAVE');
  writeString(12, 'fmt ');
  view.setUint32(16, 16, true);
  view.setUint16(20, 1, true);
  view.setUint16(22, 1, true);
  view.setUint32(24, sampleRate, true);
  view.setUint32(28, sampleRate * blockAlign, true);
  view.setUint16(32, blockAlign, true);
  view.setUint16(34, 16, true);
  writeString(36, 'data');
  view.setUint32(40, samples.length * bytesPerSample, true);

  let offset = 44;
  for (let i = 0; i < samples.length; i += 1) {
    const s = Math.max(-1, Math.min(1, samples[i]));
    view.setInt16(offset, s < 0 ? s * 0x8000 : s * 0x7fff, true);
    offset += 2;
  }
  return new Blob([view], { type: 'audio/wav' });
}

function startWavRecorder() {
  state.recorderBuffer = [];
  state.recordingSampleRate = audioCtx.sampleRate;
  const node = audioCtx.createScriptProcessor(4096, 1, 1);
  const silentGain = audioCtx.createGain();
  silentGain.gain.value = 0;
  node.onaudioprocess = (event) => {
    if (state.recordingState !== 'recording') return;
    const input = event.inputBuffer.getChannelData(0);
    state.recorderBuffer.push(new Float32Array(input));
  };
  masterGain.connect(node);
  node.connect(silentGain);
  silentGain.connect(audioCtx.destination);
  state.recorderNode = node;
  state.recorderSilentGain = silentGain;
}

function finalizeWavRecording() {
  if (!state.recorderBuffer.length) return;
  const total = state.recorderBuffer.reduce((sum, chunk) => sum + chunk.length, 0);
  const merged = new Float32Array(total);
  let offset = 0;
  for (const chunk of state.recorderBuffer) {
    merged.set(chunk, offset);
    offset += chunk.length;
  }
  const blob = encodeWav(merged, state.recordingSampleRate || 44100);
  if (els.downloadLink.href) URL.revokeObjectURL(els.downloadLink.href);
  if (els.playback?.src) URL.revokeObjectURL(els.playback.src);
  if (els.playback?.src) URL.revokeObjectURL(els.playback.src);
  const url = URL.createObjectURL(blob);
  els.downloadLink.href = url;
  els.downloadLink.download = `glance-note-${Date.now()}.wav`;
  els.downloadLink.classList.remove('hidden');
  if (els.playback) {
    els.playback.src = url;
    els.playback.classList.remove('hidden');
    els.playback.load();
  }
}

function cleanupWavRecorder() {
  try { state.recorderNode?.disconnect(); } catch (_) {}
  try { state.recorderSilentGain?.disconnect(); } catch (_) {}
  state.recorderNode = null;
  state.recorderSilentGain = null;
}


function updateRecordingUI() {
  const mode = state.recordingState;
  els.recIndicator.className = 'rec-indicator';
  els.downloadLink.classList.toggle('hidden', !els.downloadLink.href);
  if (els.playback) els.playback.classList.toggle('hidden', !els.downloadLink.href);

  if (mode === 'pending') {
    els.recIndicator.classList.add('pending');
    els.recordButton.textContent = 'Cancel';
    els.recordButton.setAttribute('aria-pressed', 'true');
  } else if (mode === 'recording') {
    els.recIndicator.classList.add('recording');
    els.recordButton.textContent = 'Stop';
    els.recordButton.setAttribute('aria-pressed', 'true');
  } else {
    els.recordButton.textContent = 'Record';
    els.recordButton.setAttribute('aria-pressed', 'false');
  }
}

async function beginRecording() {
  if (!state.audioReady) setupAudio();
  if (audioCtx.state === 'suspended') await audioCtx.resume();

  cleanupWavRecorder();
  state.recorderBuffer = [];
  startWavRecorder();
  state.recordingState = 'recording';
  els.countdownText.textContent = '';
  updateRecordingUI();
  setMessage('Recording… move gently. Double blink to change voice.');
  pulse();
}

function queueRecording() {
  if (state.pendingRecordTimer) clearInterval(state.pendingRecordTimer);
  let remaining = 0.8;
  state.recordingState = 'pending';
  updateRecordingUI();
  els.countdownText.textContent = `REC in ${remaining.toFixed(1)}s`;
  setMessage('Get ready… recording starts in 0.8 seconds.');

  const startAt = performance.now() + 800;
  state.pendingRecordTimer = setInterval(() => {
    const msLeft = startAt - performance.now();
    if (msLeft <= 0) {
      clearInterval(state.pendingRecordTimer);
      state.pendingRecordTimer = null;
      beginRecording();
      return;
    }
    remaining = Math.max(0, msLeft / 1000);
    els.countdownText.textContent = `REC in ${remaining.toFixed(1)}s`;
  }, 60);
}

function cancelQueuedRecording() {
  if (state.pendingRecordTimer) {
    clearInterval(state.pendingRecordTimer);
    state.pendingRecordTimer = null;
  }
  state.recordingState = 'idle';
  els.countdownText.textContent = '';
  updateRecordingUI();
  setMessage('Recording cancelled.');
}

function stopRecording() {
  if (state.pendingRecordTimer) {
    cancelQueuedRecording();
    return;
  }
  if (state.recordingState === 'recording') {
    state.recordingState = 'idle';
    cleanupWavRecorder();
    finalizeWavRecording();
  }
  updateRecordingUI();
  els.countdownText.textContent = 'Saved below.';
  setMessage('Recording stopped. Saved as WAV.');
}

function cycleVoice() {
  state.voiceIndex = (state.voiceIndex + 1) % state.voices.length;
  if (osc) osc.type = state.voices[state.voiceIndex];
  updateVoiceLabel();
  pulse();
}

function maybeHandleDoubleBlink() {
  const now = performance.now();
  state.blinkEvents = state.blinkEvents.filter((t) => now - t < 900);
  if (state.blinkEvents.length >= 2 && now - state.lastBlinkHandledAt > 500) {
    const lastTwo = state.blinkEvents.slice(-2);
    const gap = lastTwo[1] - lastTwo[0];
    if (gap >= 80 && gap <= 900) {
      state.lastBlinkHandledAt = now;
      state.blinkEvents = [];
      cycleVoice();
      setMessage(`Voice changed to ${state.voices[state.voiceIndex].replace('sawtooth', 'saw')}.`);
    }
  }
}

async function setupCamera() {
  const constraints = {
    audio: false,
    video: {
      facingMode: 'user',
      width: { ideal: 1280 },
      height: { ideal: 720 },
    },
  };

  state.stream = await navigator.mediaDevices.getUserMedia(constraints);
  els.video.srcObject = state.stream;
  await els.video.play();
  state.cameraReady = true;
}

async function setupFaceLandmarker() {
  const vision = await FilesetResolver.forVisionTasks(WASM_URL);
  state.faceLandmarker = await FaceLandmarker.createFromOptions(vision, {
    baseOptions: {
      modelAssetPath: MODEL_URL,
      delegate: 'GPU',
    },
    runningMode: 'VIDEO',
    numFaces: 1,
    minFaceDetectionConfidence: 0.5,
    minFacePresenceConfidence: 0.5,
    minTrackingConfidence: 0.5,
  });
  state.faceReady = true;
}

function initCalibration(nx, ny) {
  state.calibration = {
    centerX: nx,
    centerY: ny,
  };
}

function updateAudioFromEyes() {
  if (!state.audioReady) return;

  const smoothingX = 0.18;
  const smoothingY = 0.28;
  state.smoothX += (state.normalizedX - state.smoothX) * smoothingX;
  state.smoothY += (state.normalizedY - state.smoothY) * smoothingY;

  const pitchMin = 110;
  const pitchMax = 880;
  const freq = pitchMin * Math.pow(pitchMax / pitchMin, state.smoothX);
  const volume = clamp(1 - state.smoothY, 0, 1);
  const filterValue = 700 + (1 - state.smoothY) * 2500;

  const now = audioCtx?.currentTime || 0;
  if (osc) osc.frequency.setTargetAtTime(freq, now, 0.035);
  if (outputGain) outputGain.gain.setTargetAtTime(volume * 0.22, now, 0.05);
  if (filterNode) filterNode.frequency.setTargetAtTime(filterValue, now, 0.06);
}

function softenWhenNoFace() {
  if (!state.audioReady || !outputGain || !audioCtx) return;
  outputGain.gain.setTargetAtTime(0.0, audioCtx.currentTime, 0.12);
}

function processFaceResult(result) {
  const faces = result.faceLandmarks || [];
  if (!faces.length) {
    const staleFor = performance.now() - state.lastFaceSeenAt;
    if (staleFor > 250) {
      setAppState('searching');
      softenWhenNoFace();
    }
    return;
  }

  const landmarks = faces[0];
  const leftIris = irisRelative(landmarks, EYE.left);
  const rightIris = irisRelative(landmarks, EYE.right);

  if (!leftIris || !rightIris) {
    softenWhenNoFace();
    setAppState('searching');
    setMessage('Eyes not detected clearly. Hold the phone a little closer.');
    return;
  }

  const irisX = (leftIris.x + rightIris.x) * 0.5;
  const irisY = (leftIris.y + rightIris.y) * 0.5;

  if (!state.calibration) initCalibration(irisX, irisY);

  const ref = state.calibration;
  const xRange = 0.18;
  const yRange = 0.07;
  const offsetX = clamp((irisX - ref.centerX) / xRange, -1, 1);
  const rawOffsetY = clamp((irisY - ref.centerY) / yRange, -1, 1);
  const verticalGain = 2.8;
  const curvedOffsetY = Math.sign(rawOffsetY) * Math.pow(Math.abs(rawOffsetY), 0.72);
  const offsetY = clamp(curvedOffsetY * verticalGain, -1, 1);

  state.normalizedX = (offsetX + 1) / 2;
  state.normalizedY = (offsetY + 1) / 2;
  state.lastFaceSeenAt = performance.now();
  setAppState('live');
  updateAudioFromEyes();

  const leftEAR = computeEAR(landmarks, EYE.left);
  const rightEAR = computeEAR(landmarks, EYE.right);
  const earRaw = (leftEAR + rightEAR) * 0.5;

  if (state.earSmoothed == null) state.earSmoothed = earRaw;
  state.earSmoothed += (earRaw - state.earSmoothed) * 0.5;
  const ear = state.earSmoothed;

  if (state.earBaseline == null) state.earBaseline = ear;
  // Adapt baseline only while clearly open, to avoid learning a half-closed eye.
  if (!state.eyeClosed && ear > state.earBaseline * 0.85) {
    state.earBaseline += (ear - state.earBaseline) * 0.02;
  }

  const baseline = state.earBaseline || ear || 0.3;
  const closeThreshold = Math.max(0.16, baseline * 0.88);
  const openThreshold = Math.max(closeThreshold + 0.015, baseline * 0.95);
  const now = performance.now();

  if (!state.eyeClosed && ear < closeThreshold) {
    state.eyeClosed = true;
    state.eyeClosedAt = now;
  } else if (state.eyeClosed && ear > openThreshold) {
    const closedFor = now - (state.eyeClosedAt || now);
    state.eyeClosed = false;
    state.eyeClosedAt = 0;
    if (closedFor >= 15 && closedFor <= 550) {
      state.blinkEvents.push(now);
      maybeHandleDoubleBlink();
      pulse();
    }
  }

  const debugVoice = state.voices[state.voiceIndex].replace('sawtooth', 'saw');
  const blinkState = state.eyeClosed ? 'closed' : 'open';
  els.message.textContent = `Ready. Gaze X ${state.normalizedX.toFixed(2)} Y ${state.normalizedY.toFixed(2)} · Y gain 2.8 · EAR ${ear.toFixed(3)} / ${baseline.toFixed(3)} · th ${closeThreshold.toFixed(3)} · ${blinkState} · ${debugVoice}`;
}

function renderLoop() {
  if (!state.running) return;
  if (state.faceLandmarker && els.video.readyState >= 2) {
    const nowMs = performance.now();
    if (els.video.currentTime !== state.lastVideoTime) {
      const result = state.faceLandmarker.detectForVideo(els.video, nowMs);
      processFaceResult(result);
      state.lastVideoTime = els.video.currentTime;
    }
  }
  animationHandle = requestAnimationFrame(renderLoop);
}

async function boot() {
  updateVoiceLabel();
  updateRecordingUI();
  if ('serviceWorker' in navigator) {
    try {
      const reg = await navigator.serviceWorker.register('./sw.js?v=5', { updateViaCache: 'none' });
      setSwState(reg.active ? 'active' : 'registered');
      navigator.serviceWorker.ready.then(() => setSwState('ready')).catch(() => {});
    } catch (err) {
      console.error(err);
      setSwState('error');
    }
  } else {
    setSwState('unsupported');
  }

  els.recordButton.addEventListener('click', async () => {
    try {
      if (!state.audioReady) setupAudio();
      if (audioCtx.state === 'suspended') await audioCtx.resume();

      if (state.recordingState === 'idle') {
        queueRecording();
      } else if (state.recordingState === 'pending') {
        cancelQueuedRecording();
      } else if (state.recordingState === 'recording') {
        stopRecording();
      }
    } catch (error) {
      console.error(error);
      setMessage('Audio could not be started on this device.');
    }
  });

  try {
    setAppState('booting');
    setMessage('Loading camera and face tracking…');
    await setupCamera();
    await setupFaceLandmarker();
    setAppState('ready');
    setMessage('Ready. Keep your face near center and move your eyes gently. Double blink changes voice.');
    state.running = true;
    renderLoop();
  } catch (error) {
    console.error(error);
    setAppState('error');
    setMessage('Camera or face tracking could not start. Please use HTTPS and allow the front camera.');
  }
}

window.addEventListener('visibilitychange', () => {
  if (document.hidden) {
    softenWhenNoFace();
  }
});

window.addEventListener('beforeunload', () => {
  cancelAnimationFrame(animationHandle);
  if (els.downloadLink.href) URL.revokeObjectURL(els.downloadLink.href);
  if (els.playback?.src) URL.revokeObjectURL(els.playback.src);
  cleanupWavRecorder();
  state.stream?.getTracks().forEach((track) => track.stop());
  state.faceLandmarker?.close?.();
  audioCtx?.close?.();
});

boot();
