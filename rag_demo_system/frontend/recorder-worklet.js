class RecorderWorklet extends AudioWorkletProcessor {
  constructor() {
    super();
    this.targetSampleRate = 24000;
    // Energy-based barge-in detection
    this._energyBaseline = 0;
    this._energyAlpha = 0.02; // slow-moving average for baseline
    this._bargeInThreshold = 3.0; // current energy must exceed baseline by this factor
    this._assistantSpeaking = false;
    this._bargeInSent = false;
    this.port.onmessage = (e) => {
      if (e.data && e.data.type === "assistantSpeaking") {
        this._assistantSpeaking = e.data.value;
        if (e.data.value) {
          this._bargeInSent = false;
          this._energyBaseline = 0; // reset baseline at start of response
        }
      }
    };
  }

  process(inputs) {
    const input = inputs[0][0];
    if (!input) return true;

    // Compute RMS energy
    let sum = 0;
    for (let i = 0; i < input.length; i++) {
      sum += input[i] * input[i];
    }
    const rms = Math.sqrt(sum / input.length);

    // Downsample to target rate
    const srcRate = sampleRate;
    const ratio = srcRate / this.targetSampleRate;
    const outLength = Math.floor(input.length / ratio);
    const out = new Int16Array(outLength);

    for (let i = 0; i < outLength; i++) {
      const srcIndex = Math.floor(i * ratio);
      const s = Math.max(-1, Math.min(1, input[srcIndex]));
      out[i] = s * 0x7fff;
    }

    this.port.postMessage(out);

    // Barge-in detection during assistant speech
    if (this._assistantSpeaking && !this._bargeInSent) {
      // Update baseline (running average of echo energy)
      if (this._energyBaseline === 0) {
        this._energyBaseline = rms + 0.001; // seed with first frame + small offset
      } else {
        this._energyBaseline = this._energyBaseline * (1 - this._energyAlpha) + rms * this._energyAlpha;
      }
      // Detect spike above baseline (user speaking over echo)
      if (rms > this._energyBaseline * this._bargeInThreshold && rms > 0.01) {
        this._bargeInSent = true;
        this.port.postMessage({ type: "bargeIn", energy: rms, baseline: this._energyBaseline });
      }
    }

    return true;
  }
}

registerProcessor("recorder-worklet", RecorderWorklet);
