/**
 * Audio Worklet Processor
 * Runs in a separate thread for low-latency audio processing
 */

class AudioProcessor extends AudioWorkletProcessor {
  constructor() {
    super();
    this.bufferSize = 2048;
    this.buffer = [];
    this.bufferStartTime = null; // Track when we started filling this buffer
  }

  process(inputs, outputs, parameters) {
    const input = inputs[0];

    if (input && input.length > 0) {
      const channelData = input[0]; // Mono channel

      // Record when we start filling a new buffer
      if (this.buffer.length === 0) {
        this.bufferStartTime = currentTime; // AudioWorklet global: current audio context time
      }

      // Accumulate samples
      for (let i = 0; i < channelData.length; i++) {
        this.buffer.push(channelData[i]);
      }

      // When buffer is full, send to main thread with capture timestamp
      while (this.buffer.length >= this.bufferSize) {
        const chunk = this.buffer.splice(0, this.bufferSize);
        this.port.postMessage({
          type: 'audio',
          samples: new Float32Array(chunk),
          captureTime: this.bufferStartTime // When this audio was captured
        });
        // Reset for next buffer
        this.bufferStartTime = this.buffer.length > 0 ? currentTime : null;
      }
    }

    return true; // Keep processor alive
  }
}

registerProcessor('audio-processor', AudioProcessor);
