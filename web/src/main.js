/**
 * ChordFormer - Real-Time Chord Detection
 * Main application orchestrator
 */

import { AudioCapture } from './audioCapture.js';
import { FeatureExtractor } from './features.js';
import { ChordModel } from './model.js';
import { ChordDisplay } from './chordDisplay.js';
import { ChordSynth } from './synth.js';

class ChordDetector {
  constructor() {
    this.display = null;
    this.audioCapture = null;
    this.featureExtractor = null;
    this.model = null;
    this.synth = null;

    // Configuration
    this.config = null;
    this.normalization = null;

    // Frame buffer for temporal context
    this.frameBuffer = [];
    this.frameRmsBuffer = []; // Track RMS of each frame for silence detection
    this.bufferSize = 3; // Minimal context (~0.28 sec)
    this.hopLength = 1024; // Doubled from 512, target ~21.5 FPS
    this.sampleRate = 22050;

    // Audio buffer for feature extraction
    // Need enough samples for FFT (8192) plus sliding window
    // Use Float32Array as circular buffer for efficiency
    this.maxAudioBufferSize = 8192 * 4; // Max 4x FFT size before we start dropping
    this.audioBuffer = new Float32Array(this.maxAudioBufferSize);
    this.audioBufferWritePos = 0;
    this.audioBufferReadPos = 0;
    this.audioBufferLength = 0;

    // Track capture time of oldest unprocessed audio
    this.oldestAudioCaptureTime = null;

    this.fftSize = 4096;
    this.samplesPerFrame = this.hopLength; // Slide by hop_length each frame

    // Pre-allocated buffer to reduce garbage collection
    this.frameSamplesBuffer = new Float32Array(this.fftSize);

    // State
    this.isRunning = false;
    this.inferenceInProgress = false;

    // Silence detection threshold (RMS below this = no chord)
    this.silenceThreshold = 0.001;

    // ===== SYNTH SMOOTHING CONFIG =====
    // Separate smoothing for synth to prevent sudden chord changes
    this.synthCurrentChord = null;           // Currently playing chord on synth
    this.synthPendingChord = null;           // Chord waiting to be confirmed
    this.synthConsecutiveCount = 0;          // Count of consecutive predictions for pending chord
    this.synthRequiredForNewChord = 2;       // Predictions needed for different root
    this.synthRequiredForQualityChange = 4;  // Predictions needed for same root, different quality
    // ===== END SYNTH SMOOTHING CONFIG =====

    // Latency tracking - now uses actual audio capture times
    this.frameCaptureTimestamps = []; // Audio capture time for each frame (when audio was recorded)
    this.latencyDisplay = {
      buffer: document.getElementById('bufferLatency'),
      inference: document.getElementById('inferenceLatency'),
      total: document.getElementById('totalLatency'),
      fps: document.getElementById('fpsCounter')
    };

    // FPS tracking
    this.frameCount = 0;
    this.lastFpsUpdate = performance.now();
    this.currentFps = 0;
  }

  /**
   * Initialize the application
   */
  async init() {
    // Initialize display
    this.display = new ChordDisplay();
    this.display.setLoading('Loading model...');

    try {
      // Load configuration
      this.display.setLoading('Loading configuration...');
      await this.loadConfig();

      // Initialize feature extractor
      this.featureExtractor = new FeatureExtractor(this.config, this.normalization);

      // Load model
      this.display.setLoading('Loading ChordFormer model...');
      this.model = new ChordModel(this.config);
      await this.model.load('/model/chord_model.onnx');

      // Initialize synth
      this.synth = new ChordSynth();
      await this.synth.init();

      // Set up UI event handlers
      this.setupEventHandlers();

      // Ready to go
      this.display.setReady();
      console.log('ChordFormer initialized successfully');

    } catch (error) {
      console.error('Initialization failed:', error);
      this.display.showError(`Failed to initialize: ${error.message}`);
    }
  }

  /**
   * Load configuration files
   */
  async loadConfig() {
    // Load model config
    const configResponse = await fetch('/model/model_config.json');
    if (!configResponse.ok) {
      throw new Error('Failed to load model_config.json');
    }
    this.config = await configResponse.json();

    // Load normalization stats
    const normResponse = await fetch('/model/normalization.json');
    if (!normResponse.ok) {
      throw new Error('Failed to load normalization.json');
    }
    this.normalization = await normResponse.json();

    console.log('Configuration loaded:', this.config);
  }

  /**
   * Set up UI event handlers
   */
  setupEventHandlers() {
    const startBtn = document.getElementById('startBtn');
    const stopBtn = document.getElementById('stopBtn');
    const synthToggle = document.getElementById('synthToggle');
    const synthVolume = document.getElementById('synthVolume');

    startBtn.addEventListener('click', () => this.start());
    stopBtn.addEventListener('click', () => this.stop());

    // Synth controls
    synthToggle.addEventListener('change', (e) => {
      this.synth.setEnabled(e.target.checked);
    });

    synthVolume.addEventListener('input', (e) => {
      this.synth.setVolume(e.target.value / 100);
    });
  }

  /**
   * Start chord detection
   */
  async start() {
    if (this.isRunning) {
      return;
    }

    try {
      this.display.setListening();

      // Initialize audio capture
      this.audioCapture = new AudioCapture({
        sampleRate: this.sampleRate,
        bufferSize: 2048,
        onAudioData: (samples, captureTimeMs) => this.handleAudioData(samples, captureTimeMs)
      });

      await this.audioCapture.start();
      this.isRunning = true;

      // Clear buffers
      this.frameBuffer = [];
      this.frameRmsBuffer = [];
      this.audioBufferWritePos = 0;
      this.audioBufferReadPos = 0;
      this.audioBufferLength = 0;
      this.oldestAudioCaptureTime = null;
      this.frameCaptureTimestamps = [];

      // Reset FPS counter
      this.frameCount = 0;
      this.lastFpsUpdate = performance.now();
      this.latencyDisplay.fps.textContent = '-';

      // Warn if tab goes to background (browsers throttle background tabs)
      document.addEventListener('visibilitychange', () => {
        if (document.hidden) {
          console.warn('[Performance] Tab is in background - audio processing may be throttled!');
        } else {
          console.log('[Performance] Tab is visible again');
        }
      });

      console.log('Chord detection started');

    } catch (error) {
      console.error('Failed to start:', error);

      if (error.name === 'NotAllowedError') {
        this.display.showError('Microphone access denied. Please allow microphone access and try again.');
      } else {
        this.display.showError(`Failed to start: ${error.message}`);
      }

      this.display.setReady();
    }
  }

  /**
   * Stop chord detection
   */
  stop() {
    if (!this.isRunning) {
      return;
    }

    if (this.audioCapture) {
      this.audioCapture.stop();
    }

    this.isRunning = false;
    this.display.setReady();

    // Stop synth and reset smoothing state
    this.synth.stopChord();
    this.synthCurrentChord = null;
    this.synthPendingChord = null;
    this.synthConsecutiveCount = 0;

    console.log('Chord detection stopped');
  }

  /**
   * Handle incoming audio data
   */
  handleAudioData(samples, captureTimeMs) {

    // Track capture time of oldest unprocessed audio
    if (this.oldestAudioCaptureTime === null) {
      this.oldestAudioCaptureTime = captureTimeMs;
    }

    // Check if buffer is getting too full (falling behind)
    if (this.audioBufferLength + samples.length > this.maxAudioBufferSize) {
      // Drop oldest audio to catch up
      const toDrop = this.audioBufferLength + samples.length - this.maxAudioBufferSize + this.fftSize;
      this.audioBufferReadPos = (this.audioBufferReadPos + toDrop) % this.maxAudioBufferSize;
      this.audioBufferLength -= toDrop;
      // Update oldest capture time estimate (we dropped some audio)
      const droppedSeconds = toDrop / this.sampleRate;
      this.oldestAudioCaptureTime += droppedSeconds * 1000;
      console.warn(`[Audio] Dropping ${toDrop} samples to catch up`);
    }

    // Write samples to circular buffer
    for (let i = 0; i < samples.length; i++) {
      this.audioBuffer[this.audioBufferWritePos] = samples[i];
      this.audioBufferWritePos = (this.audioBufferWritePos + 1) % this.maxAudioBufferSize;
    }
    this.audioBufferLength += samples.length;

    // Process frames when we have enough samples for FFT
    while (this.audioBufferLength >= this.fftSize) {
      // Get the capture time of the oldest sample in this frame
      const frameCaptureTime = this.oldestAudioCaptureTime;

      // Extract fftSize samples for this frame from circular buffer (reuse pre-allocated buffer)
      const frameSamples = this.frameSamplesBuffer;
      for (let i = 0; i < this.fftSize; i++) {
        const idx = (this.audioBufferReadPos + i) % this.maxAudioBufferSize;
        frameSamples[i] = this.audioBuffer[idx];
      }

      // Slide read position by hopLength (not fftSize) for overlap
      this.audioBufferReadPos = (this.audioBufferReadPos + this.hopLength) % this.maxAudioBufferSize;
      this.audioBufferLength -= this.hopLength;

      // Update oldest capture time (we consumed hopLength samples)
      const consumedSeconds = this.hopLength / this.sampleRate;
      this.oldestAudioCaptureTime += consumedSeconds * 1000;

      // Extract raw CQT features (before normalization) for debugging
      const rawFeatures = this.featureExtractor.extractFrame(frameSamples);

      // Normalize features
      const features = this.featureExtractor.normalize(rawFeatures);

      // Store the audio capture time for this frame (when the audio was actually recorded)
      this.frameCaptureTimestamps.push(frameCaptureTime);


      // Calculate RMS of this frame's audio for silence detection
      let frameRms = 0;
      for (let i = 0; i < frameSamples.length; i++) {
        frameRms += frameSamples[i] * frameSamples[i];
      }
      frameRms = Math.sqrt(frameRms / frameSamples.length);

      // Add to frame buffer
      this.frameBuffer.push(features);
      this.frameRmsBuffer.push(frameRms);

      // Update FPS counter
      this.frameCount++;
      const now = performance.now();
      if (now - this.lastFpsUpdate >= 1000) {
        this.currentFps = this.frameCount;
        this.frameCount = 0;
        this.lastFpsUpdate = now;
        this.latencyDisplay.fps.textContent = this.currentFps;
        // Target is ~10.75 FPS (22050/2048), warn if below
        if (this.currentFps < 9) {
          this.latencyDisplay.fps.classList.add('warning');
        } else {
          this.latencyDisplay.fps.classList.remove('warning');
        }
      }

      // Keep buffer at fixed size
      if (this.frameBuffer.length > this.bufferSize) {
        this.frameBuffer.shift();
        this.frameRmsBuffer.shift();
        this.frameCaptureTimestamps.shift();
      }

      // Run inference when buffer is full and not already processing
      if (this.frameBuffer.length >= this.bufferSize && !this.inferenceInProgress) {
        this.runInference();
      }
    }
  }

  /**
   * Run model inference on buffered frames
   */
  async runInference() {
    if (this.inferenceInProgress || this.frameBuffer.length < this.bufferSize) {
      return;
    }

    this.inferenceInProgress = true;

    try {
      // Check for silence - average RMS across all frames in buffer
      const avgRms = this.frameRmsBuffer.reduce((a, b) => a + b, 0) / this.frameRmsBuffer.length;

      if (avgRms < this.silenceThreshold) {
        // Silent - show no chord (with smoothing)
        this.updateDisplaySmoothed('N', 0);
        this.inferenceInProgress = false;
        return;
      }

      // Get current frames (copy to avoid race conditions)
      const frames = [...this.frameBuffer];

      // Get the AUDIO CAPTURE TIME of the oldest frame (when audio was actually recorded)
      // This is the true start time for latency calculation
      const oldestAudioCaptureTime = this.frameCaptureTimestamps[0];
      const inferenceStartTime = performance.now();

      // Calculate pipeline latency (time from audio capture to now)
      const pipelineLatency = inferenceStartTime - oldestAudioCaptureTime;

      // Run inference
      const result = await this.model.predict(frames);

      const inferenceEndTime = performance.now();
      const inferenceLatency = inferenceEndTime - inferenceStartTime;
      const totalLatency = inferenceEndTime - oldestAudioCaptureTime;

      // Update latency display
      this.updateLatencyDisplay(pipelineLatency, inferenceLatency, totalLatency);

      // Log prediction
      console.log(`[Prediction] ${result.chord} (${(result.confidence * 100).toFixed(0)}%)`);

      // Update display with smoothing (strip slash chords for cleaner UI)
      const displayChord = this.model.stripSlashChord(result.chord);
      this.updateDisplaySmoothed(displayChord, result.confidence);

    } catch (error) {
      console.error('Inference error:', error);
    } finally {
      this.inferenceInProgress = false;
    }
  }

  /**
   * Update display with smoothing to prevent flickering
   * Simple approach: ignore low-confidence predictions
   */
  updateDisplaySmoothed(chord, confidence) {
    const minConfidence = 0.4; // Ignore predictions below 40%

    // If confidence is too low, don't update display at all
    if (confidence < minConfidence) {
      return;
    }

    // High enough confidence - update display
    this.display.updateChord(chord, confidence);

    // Update synth with separate smoothing logic
    this.updateSynthSmoothed(chord);
  }

  // ===== SYNTH SMOOTHING METHODS =====

  /**
   * Parse chord string to extract root and quality
   * e.g., "A:maj" -> { root: "A", quality: "maj" }
   */
  parseChordParts(chord) {
    if (!chord || chord === 'N' || chord === '-') {
      return { root: null, quality: null };
    }

    if (chord.includes(':')) {
      const [root, quality] = chord.split(':');
      return { root, quality };
    }

    return { root: chord, quality: 'maj' };
  }

  /**
   * Check if two chords have the same root but different quality
   */
  isSameRootDifferentQuality(chord1, chord2) {
    const parts1 = this.parseChordParts(chord1);
    const parts2 = this.parseChordParts(chord2);

    return parts1.root !== null &&
           parts2.root !== null &&
           parts1.root === parts2.root &&
           parts1.quality !== parts2.quality;
  }

  /**
   * Update synth with smoothing to prevent sudden chord changes
   * Requires multiple consecutive predictions before changing
   */
  updateSynthSmoothed(chord) {
    // If same as currently playing, do nothing
    if (chord === this.synthCurrentChord) {
      this.synthPendingChord = null;
      this.synthConsecutiveCount = 0;
      return;
    }

    // Check if this is the same pending chord
    if (chord === this.synthPendingChord) {
      this.synthConsecutiveCount++;
    } else {
      // New pending chord
      this.synthPendingChord = chord;
      this.synthConsecutiveCount = 1;
    }

    // Determine required count based on type of change
    let requiredCount;
    if (this.isSameRootDifferentQuality(this.synthCurrentChord, chord)) {
      // Same root, different quality (e.g., Amaj -> Amin) - need more predictions
      requiredCount = this.synthRequiredForQualityChange;
    } else {
      // Different root - need fewer predictions
      requiredCount = this.synthRequiredForNewChord;
    }

    // Check if we have enough consecutive predictions
    if (this.synthConsecutiveCount >= requiredCount) {
      // Confirmed - update synth
      this.synthCurrentChord = chord;
      this.synth.playChord(chord);
      this.synthPendingChord = null;
      this.synthConsecutiveCount = 0;
    }
  }

  // ===== END SYNTH SMOOTHING METHODS =====

  /**
   * Update latency display on the UI
   */
  updateLatencyDisplay(bufferMs, inferenceMs, totalMs) {
    const formatMs = (ms) => {
      if (ms >= 1000) {
        return `${(ms / 1000).toFixed(1)}s`;
      }
      return `${Math.round(ms)}ms`;
    };

    this.latencyDisplay.buffer.textContent = formatMs(bufferMs);
    this.latencyDisplay.inference.textContent = formatMs(inferenceMs);
    this.latencyDisplay.total.textContent = formatMs(totalMs);

    // Add warning class if total latency is high
    if (totalMs > 2000) {
      this.latencyDisplay.total.classList.add('warning');
    } else {
      this.latencyDisplay.total.classList.remove('warning');
    }
  }
}

// Initialize when DOM is ready
document.addEventListener('DOMContentLoaded', () => {
  const detector = new ChordDetector();
  detector.init();
});
