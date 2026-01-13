/**
 * Audio Capture Module
 * Handles microphone input using Web Audio API with AudioWorklet
 */

export class AudioCapture {
  constructor(options = {}) {
    this.targetSampleRate = options.sampleRate || 22050;
    this.onAudioData = options.onAudioData || null;

    this.audioContext = null;
    this.sourceNode = null;
    this.workletNode = null;
    this.stream = null;
    this.isCapturing = false;

    // For converting AudioContext time to performance.now() time
    this.audioContextStartTime = null; // AudioContext.currentTime when we started
    this.performanceStartTime = null;  // performance.now() when we started
  }

  /**
   * Initialize audio capture from microphone
   */
  async start() {
    if (this.isCapturing) {
      return;
    }

    try {
      // Request microphone access
      this.stream = await navigator.mediaDevices.getUserMedia({
        audio: {
          echoCancellation: false,
          noiseSuppression: false,
          autoGainControl: false,
          channelCount: 1
        }
      });

      // Create audio context
      // Note: Browser may use a different sample rate (44100 or 48000)
      // We'll resample to target rate
      this.audioContext = new (window.AudioContext || window.webkitAudioContext)();
      this.actualSampleRate = this.audioContext.sampleRate;

      console.log(`Browser audio sample rate: ${this.actualSampleRate} Hz`);
      console.log(`Target sample rate: ${this.targetSampleRate} Hz`);

      // Load and register the audio worklet processor
      await this.audioContext.audioWorklet.addModule('/audio-processor.js');

      // Create source from microphone stream
      this.sourceNode = this.audioContext.createMediaStreamSource(this.stream);

      // Create worklet node
      this.workletNode = new AudioWorkletNode(this.audioContext, 'audio-processor');

      // Record time sync for converting AudioContext time to performance.now()
      this.audioContextStartTime = this.audioContext.currentTime;
      this.performanceStartTime = performance.now();

      // Handle messages from the worklet
      this.workletNode.port.onmessage = (event) => {
        if (event.data.type === 'audio') {
          // Convert AudioContext time to performance.now() time
          const captureTimeMs = this.audioContextTimeToPerformance(event.data.captureTime);
          this.handleAudioData(event.data.samples, captureTimeMs);
        }
      };

      // Connect: microphone -> worklet
      this.sourceNode.connect(this.workletNode);
      // Note: We don't connect to destination to avoid feedback

      this.isCapturing = true;
      console.log('Audio capture started (using AudioWorklet)');

    } catch (error) {
      console.error('Failed to start audio capture:', error);
      throw error;
    }
  }

  /**
   * Convert AudioContext time (seconds) to performance.now() time (milliseconds)
   */
  audioContextTimeToPerformance(audioContextTime) {
    if (audioContextTime === null || audioContextTime === undefined) {
      return performance.now(); // Fallback
    }
    const elapsedAudioTime = audioContextTime - this.audioContextStartTime;
    return this.performanceStartTime + (elapsedAudioTime * 1000);
  }

  /**
   * Handle audio data from worklet
   */
  handleAudioData(samples, captureTimeMs) {
    // Resample if necessary
    let processedSamples;
    if (this.actualSampleRate !== this.targetSampleRate) {
      processedSamples = this.resample(samples, this.actualSampleRate, this.targetSampleRate);
    } else {
      processedSamples = samples;
    }

    // Send to callback with capture time
    if (this.onAudioData) {
      this.onAudioData(processedSamples, captureTimeMs);
    }
  }

  /**
   * Simple linear interpolation resampling
   */
  resample(inputSamples, fromRate, toRate) {
    const ratio = fromRate / toRate;
    const outputLength = Math.floor(inputSamples.length / ratio);
    const output = new Float32Array(outputLength);

    for (let i = 0; i < outputLength; i++) {
      const srcIndex = i * ratio;
      const srcIndexFloor = Math.floor(srcIndex);
      const srcIndexCeil = Math.min(srcIndexFloor + 1, inputSamples.length - 1);
      const t = srcIndex - srcIndexFloor;

      output[i] = inputSamples[srcIndexFloor] * (1 - t) + inputSamples[srcIndexCeil] * t;
    }

    return output;
  }

  /**
   * Stop audio capture
   */
  stop() {
    if (!this.isCapturing) {
      return;
    }

    // Disconnect nodes
    if (this.workletNode) {
      this.workletNode.disconnect();
      this.workletNode.port.onmessage = null;
    }
    if (this.sourceNode) {
      this.sourceNode.disconnect();
    }

    // Stop media stream tracks
    if (this.stream) {
      this.stream.getTracks().forEach(track => track.stop());
    }

    // Close audio context
    if (this.audioContext && this.audioContext.state !== 'closed') {
      this.audioContext.close();
    }

    this.isCapturing = false;
    console.log('Audio capture stopped');
  }

  /**
   * Get target sample rate
   */
  getSampleRate() {
    return this.targetSampleRate;
  }

  /**
   * Check if currently capturing
   */
  isActive() {
    return this.isCapturing;
  }
}
