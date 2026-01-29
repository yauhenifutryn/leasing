class RecorderWorklet extends AudioWorkletProcessor {
  constructor() {
    super();
    this.targetSampleRate = 24000;
    this.port.onmessage = (e) => {
      if (e.data === "reset") {
        return;
      }
    };
  }

  process(inputs) {
    const input = inputs[0][0];
    if (!input) return true;

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
    return true;
  }
}

registerProcessor("recorder-worklet", RecorderWorklet);
