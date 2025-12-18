/**
 * Audio Capture Module
 * Handles microphone input using Web Audio API with AudioWorklet for low-latency processing
 */

export class AudioCapture {
    constructor(options = {}) {
        this.sampleRate = options.sampleRate || 22050;
        this.bufferSize = options.bufferSize || 2048;
        this.onAudioData = options.onAudioData || (() => {});

        this.audioContext = null;
        this.source = null;
        this.workletNode = null;
        this.analyser = null;
        this.stream = null;

        this.isRunning = false;
    }

    async start() {
        try {
            // Request microphone access
            this.stream = await navigator.mediaDevices.getUserMedia({
                audio: {
                    echoCancellation: false,
                    noiseSuppression: false,
                    autoGainControl: false,
                    sampleRate: this.sampleRate
                }
            });

            // Create audio context with target sample rate
            this.audioContext = new (window.AudioContext || window.webkitAudioContext)({
                sampleRate: this.sampleRate,
                latencyHint: 'interactive'
            });

            // Register AudioWorklet
            await this.audioContext.audioWorklet.addModule('/audio-processor.js');

            // Create source from microphone
            this.source = this.audioContext.createMediaStreamSource(this.stream);

            // Create analyser for visualization
            this.analyser = this.audioContext.createAnalyser();
            this.analyser.fftSize = 2048;
            this.source.connect(this.analyser);

            // Create AudioWorklet node
            this.workletNode = new AudioWorkletNode(this.audioContext, 'audio-processor');

            // Handle messages from worklet
            this.workletNode.port.onmessage = (event) => {
                if (!this.isRunning) return;
                if (event.data.type === 'audio') {
                    this.onAudioData(event.data.samples);
                }
            };

            // Connect nodes
            this.source.connect(this.workletNode);
            this.workletNode.connect(this.audioContext.destination);

            this.isRunning = true;
            console.log(`Audio capture started at ${this.audioContext.sampleRate}Hz`);

        } catch (error) {
            if (error.name === 'NotAllowedError') {
                throw new Error('Microphone access denied. Please allow microphone access and try again.');
            } else if (error.name === 'NotFoundError') {
                throw new Error('No microphone found. Please connect a microphone and try again.');
            } else {
                throw new Error(`Failed to start audio capture: ${error.message}`);
            }
        }
    }

    stop() {
        this.isRunning = false;

        if (this.workletNode) {
            this.workletNode.disconnect();
            this.workletNode = null;
        }

        if (this.source) {
            this.source.disconnect();
            this.source = null;
        }

        if (this.analyser) {
            this.analyser.disconnect();
            this.analyser = null;
        }

        if (this.stream) {
            this.stream.getTracks().forEach(track => track.stop());
            this.stream = null;
        }

        if (this.audioContext) {
            this.audioContext.close();
            this.audioContext = null;
        }

        console.log('Audio capture stopped');
    }

    getActualSampleRate() {
        return this.audioContext ? this.audioContext.sampleRate : this.sampleRate;
    }
}
