/**
 * Main entry point for real-time chord detection
 */

import { AudioCapture } from './audioCapture.js';
import { FeatureExtractor } from './features.js';
import { ChordModel } from './model.js';
import { ChordDisplay } from './chordDisplay.js';
import { ChordSynth } from './chordSynth.js';

class ChordDetector {
    constructor() {
        this.audioCapture = null;
        this.featureExtractor = null;
        this.model = null;
        this.display = null;
        this.synth = null;

        this.isRunning = false;
        this.frameBuffer = [];
        this.bufferSize = 16;

        // Performance tracking
        this.frameCount = 0;
        this.lastFpsUpdate = 0;
        this.inferenceLatencies = [];

        // UI elements
        this.startBtn = document.getElementById('startBtn');
        this.stopBtn = document.getElementById('stopBtn');
        this.synthBtn = document.getElementById('synthBtn');
        this.volumeSlider = document.getElementById('volumeSlider');
        this.volumeControl = document.getElementById('volumeControl');
        this.statusDot = document.getElementById('statusDot');
        this.statusText = document.getElementById('statusText');
        this.errorContainer = document.getElementById('errorContainer');

        this.init();
    }

    async init() {
        try {
            this.setStatus('loading', 'Loading model...');

            // Initialize components
            this.display = new ChordDisplay();
            this.featureExtractor = new FeatureExtractor();
            this.model = new ChordModel();
            this.synth = new ChordSynth();

            // Load model and config
            await this.model.load();
            await this.featureExtractor.loadNormalization();

            this.setStatus('ready', 'Ready - Click Start to begin');
            this.startBtn.disabled = false;

            // Set up event listeners
            this.startBtn.addEventListener('click', () => this.start());
            this.stopBtn.addEventListener('click', () => this.stop());
            this.synthBtn.addEventListener('click', () => this.toggleSynth());
            this.volumeSlider.addEventListener('input', (e) => this.updateVolume(e.target.value));

        } catch (error) {
            this.setStatus('error', 'Failed to initialize');
            this.showError(error.message);
            console.error('Initialization error:', error);
        }
    }

    async start() {
        try {
            this.setStatus('loading', 'Requesting microphone access...');

            // Initialize audio capture
            this.audioCapture = new AudioCapture({
                sampleRate: 22050,
                bufferSize: 2048,
                onAudioData: (samples) => this.processAudio(samples)
            });

            await this.audioCapture.start();

            this.isRunning = true;
            this.frameBuffer = [];
            this.frameCount = 0;
            this.lastFpsUpdate = performance.now();

            this.setStatus('listening', 'Listening...');
            this.startBtn.disabled = true;
            this.stopBtn.disabled = false;

            // Start visualization
            this.display.startVisualization(this.audioCapture.analyser);

        } catch (error) {
            this.setStatus('error', 'Failed to start');
            this.showError(error.message);
            console.error('Start error:', error);
        }
    }

    stop() {
        this.isRunning = false;

        if (this.audioCapture) {
            this.audioCapture.stop();
            this.audioCapture = null;
        }

        this.display.stopVisualization();
        this.display.updateChord('-', 'Stopped');

        this.setStatus('ready', 'Ready - Click Start to begin');
        this.startBtn.disabled = false;
        this.stopBtn.disabled = true;
    }

    async processAudio(samples) {
        if (!this.isRunning) return;

        const startTime = performance.now();

        try {
            // Extract CQT features from audio samples
            const features = this.featureExtractor.extractCQT(samples);

            // Add to frame buffer
            this.frameBuffer.push(features);

            // Keep buffer at max size
            if (this.frameBuffer.length > this.bufferSize) {
                this.frameBuffer.shift();
            }

            // Need at least a few frames for context
            if (this.frameBuffer.length < 4) {
                return;
            }

            // Stack frames into batch
            const batchFeatures = this.stackFrames(this.frameBuffer);

            // Run inference
            const result = await this.model.predict(batchFeatures);

            // Get the prediction for the most recent frame
            const lastPrediction = result.predictions[result.predictions.length - 1];
            const lastConfidence = result.confidences[result.confidences.length - 1];

            // Update display
            this.display.updateChord(lastPrediction, null, lastConfidence);

            // Play chord on synth if enabled
            if (this.synth && this.synth.isEnabled) {
                this.synth.playChord(lastPrediction);
            }

            // Track performance
            const latency = performance.now() - startTime;
            this.inferenceLatencies.push(latency);
            if (this.inferenceLatencies.length > 30) {
                this.inferenceLatencies.shift();
            }

            this.frameCount++;
            this.updateStats(latency, lastConfidence);

        } catch (error) {
            console.error('Processing error:', error);
        }
    }

    stackFrames(frames) {
        // Stack frames into a 2D array [numFrames, 84]
        const numFrames = frames.length;
        const numBins = 84;
        const stacked = new Float32Array(numFrames * numBins);

        for (let i = 0; i < numFrames; i++) {
            stacked.set(frames[i], i * numBins);
        }

        return {
            data: stacked,
            shape: [1, numFrames, numBins]
        };
    }

    updateStats(latency, confidence) {
        // Update latency display
        const avgLatency = this.inferenceLatencies.reduce((a, b) => a + b, 0) / this.inferenceLatencies.length;
        document.getElementById('latency').textContent = avgLatency.toFixed(1);

        // Update confidence
        document.getElementById('confidence').textContent = (confidence * 100).toFixed(0) + '%';

        // Update FPS
        const now = performance.now();
        if (now - this.lastFpsUpdate > 1000) {
            const fps = this.frameCount / ((now - this.lastFpsUpdate) / 1000);
            document.getElementById('fps').textContent = fps.toFixed(1);
            this.frameCount = 0;
            this.lastFpsUpdate = now;
        }
    }

    toggleSynth() {
        const enabled = this.synth.toggle();
        this.synthBtn.textContent = enabled ? 'Synth: ON' : 'Synth: OFF';
        this.synthBtn.classList.toggle('active', enabled);
        this.volumeControl.classList.toggle('active', enabled);
    }

    updateVolume(value) {
        const volume = value / 100;
        this.synth.setVolume(volume);
    }

    setStatus(state, message) {
        this.statusDot.className = 'status-dot ' + state;
        this.statusText.textContent = message;
    }

    showError(message) {
        this.errorContainer.innerHTML = `
            <div class="error-message">
                <strong>Error:</strong> ${message}
            </div>
        `;
    }
}

// Initialize when DOM is ready
document.addEventListener('DOMContentLoaded', () => {
    new ChordDetector();
});
