/* The Talk button: record the voice in the browser, send a WAV file to the server, give the words to the page.
 *
 * The recording stays in the memory of the browser. It goes to the same server only, and the server does not
 * keep it. The browser speech services are not used, because they can send the audio to a cloud service.
 * The microphone works only on a page with HTTPS (or on localhost).
 */
(function () {
  const TARGET_RATE = 16000;
  const state = {
    recording: false,
    busy: false,
    chunks: [],
    context: null,
    stream: null,
    processor: null,
    timer: null,
    maxSeconds: 15,
    endpoint: '/ui/voice',
  };

  function button() {
    return document.getElementById('saarthi-talk');
  }

  function setLabel(text, mode) {
    const el = button();
    if (!el) { return; }
    el.querySelector('.up-talk__text').textContent = text;
    el.dataset.mode = mode;
    el.setAttribute('aria-pressed', mode === 'listening' ? 'true' : 'false');
  }

  function fail(message) {
    state.busy = false;
    setLabel('Talk', 'idle');
    emitEvent('saarthi_voice_error', { message: message });
  }

  function resample(samples, fromRate) {
    if (fromRate === TARGET_RATE) { return samples; }
    const ratio = fromRate / TARGET_RATE;
    const length = Math.floor(samples.length / ratio);
    const out = new Float32Array(length);
    for (let i = 0; i < length; i++) {
      const position = i * ratio;
      const index = Math.floor(position);
      const next = Math.min(index + 1, samples.length - 1);
      const fraction = position - index;
      out[i] = samples[index] * (1 - fraction) + samples[next] * fraction;
    }
    return out;
  }

  function encodeWav(samples) {
    const buffer = new ArrayBuffer(44 + samples.length * 2);
    const view = new DataView(buffer);
    const text = (offset, value) => {
      for (let i = 0; i < value.length; i++) { view.setUint8(offset + i, value.charCodeAt(i)); }
    };
    text(0, 'RIFF');
    view.setUint32(4, 36 + samples.length * 2, true);
    text(8, 'WAVE');
    text(12, 'fmt ');
    view.setUint32(16, 16, true);
    view.setUint16(20, 1, true);            // PCM
    view.setUint16(22, 1, true);            // One channel
    view.setUint32(24, TARGET_RATE, true);
    view.setUint32(28, TARGET_RATE * 2, true);
    view.setUint16(32, 2, true);
    view.setUint16(34, 16, true);
    text(36, 'data');
    view.setUint32(40, samples.length * 2, true);
    for (let i = 0; i < samples.length; i++) {
      const value = Math.max(-1, Math.min(1, samples[i]));
      view.setInt16(44 + i * 2, value < 0 ? value * 0x8000 : value * 0x7fff, true);
    }
    return new Blob([buffer], { type: 'audio/wav' });
  }

  async function send(blob) {
    try {
      const response = await fetch(state.endpoint, {
        method: 'POST',
        body: blob,
        headers: { 'Content-Type': 'audio/wav' },
        cache: 'no-store',
      });
      const data = await response.json();
      if (!response.ok) { return fail(data.error || 'The computer could not listen.'); }
      state.busy = false;
      setLabel('Talk', 'idle');
      emitEvent('saarthi_heard', { text: data.text });
    } catch (error) {
      fail('The computer could not listen. Please try again.');
    }
  }

  function release() {
    clearTimeout(state.timer);
    if (state.processor) { state.processor.disconnect(); state.processor.onaudioprocess = null; }
    if (state.stream) { state.stream.getTracks().forEach((track) => track.stop()); }
    const rate = state.context ? state.context.sampleRate : TARGET_RATE;
    if (state.context) { state.context.close(); }
    state.processor = null; state.stream = null; state.context = null;
    return rate;
  }

  function stop() {
    if (!state.recording) { return; }
    state.recording = false;
    const rate = release();
    let total = 0;
    state.chunks.forEach((chunk) => { total += chunk.length; });
    const joined = new Float32Array(total);
    let offset = 0;
    state.chunks.forEach((chunk) => { joined.set(chunk, offset); offset += chunk.length; });
    state.chunks = [];
    state.busy = true;
    setLabel('Listening to you...', 'busy');
    send(encodeWav(resample(joined, rate)));
  }

  async function start() {
    if (!window.isSecureContext || !navigator.mediaDevices || !navigator.mediaDevices.getUserMedia) {
      return fail('The microphone needs a secure page (HTTPS).');
    }
    try {
      state.stream = await navigator.mediaDevices.getUserMedia({
        audio: { channelCount: 1, echoCancellation: true, noiseSuppression: true },
      });
    } catch (error) {
      return fail('The microphone is not allowed. Please allow it in the browser.');
    }
    const Context = window.AudioContext || window.webkitAudioContext;
    state.context = new Context();
    const source = state.context.createMediaStreamSource(state.stream);
    state.processor = state.context.createScriptProcessor(4096, 1, 1);
    state.chunks = [];
    state.processor.onaudioprocess = (event) => {
      state.chunks.push(new Float32Array(event.inputBuffer.getChannelData(0)));
    };
    source.connect(state.processor);
    state.processor.connect(state.context.destination);
    state.recording = true;
    setLabel('Stop', 'listening');
    // The answer of a long recording is refused by the server, so stop before the limit.
    state.timer = setTimeout(stop, state.maxSeconds * 1000);
  }

  window.saarthiTalk = {
    configure: function (options) {
      state.maxSeconds = options.maxSeconds;
      state.endpoint = options.endpoint;
    },
    toggle: function () {
      if (state.busy) { return; }
      if (state.recording) { stop(); } else { start(); }
    },
  };
})();
