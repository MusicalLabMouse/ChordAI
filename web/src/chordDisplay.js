/**
 * Chord Display Module
 * Handles UI updates and audio visualization
 */

export class ChordDisplay {
    constructor() {
        this.chordNameEl = document.getElementById('chordName');
        this.chordTypeEl = document.getElementById('chordType');
        this.chordHistoryEl = document.getElementById('chordHistory');
        this.canvas = document.getElementById('waveform');
        this.ctx = this.canvas.getContext('2d');

        this.currentChord = '-';
        this.chordHistory = [];
        this.maxHistory = 8;

        this.analyser = null;
        this.animationId = null;
        this.dataArray = null;

        this.setupCanvas();
    }

    setupCanvas() {
        // Set canvas size
        const rect = this.canvas.parentElement.getBoundingClientRect();
        this.canvas.width = rect.width;
        this.canvas.height = rect.height;
    }

    updateChord(chordName, type = null, confidence = null) {
        // Only update if chord changed
        if (chordName !== this.currentChord) {
            this.currentChord = chordName;

            // Add changing animation
            this.chordNameEl.classList.add('changing');
            this.chordNameEl.textContent = chordName;

            // Remove animation class after transition
            setTimeout(() => {
                this.chordNameEl.classList.remove('changing');
            }, 100);

            // Add to history
            if (chordName !== '-') {
                this.addToHistory(chordName);
            }
        }

        // Update type/status text
        if (type !== null) {
            this.chordTypeEl.textContent = type;
        } else if (confidence !== null) {
            const confPercent = (confidence * 100).toFixed(0);
            this.chordTypeEl.textContent = `Confidence: ${confPercent}%`;
        }

        // Update visual feedback based on confidence
        if (confidence !== null) {
            this.updateConfidenceVisual(confidence);
        }
    }

    updateConfidenceVisual(confidence) {
        // Adjust chord display opacity/color based on confidence
        const opacity = 0.5 + confidence * 0.5;
        const hue = 160 + (1 - confidence) * 40;  // Green to yellow
        this.chordNameEl.style.color = `hsla(${hue}, 100%, 75%, ${opacity})`;
    }

    addToHistory(chord) {
        // Don't add duplicate consecutive chords
        if (this.chordHistory.length > 0 && this.chordHistory[0] === chord) {
            return;
        }

        this.chordHistory.unshift(chord);
        if (this.chordHistory.length > this.maxHistory) {
            this.chordHistory.pop();
        }

        this.renderHistory();
    }

    renderHistory() {
        this.chordHistoryEl.innerHTML = this.chordHistory
            .map((chord, i) => `<span class="history-chord">${chord}</span>`)
            .join('');
    }

    startVisualization(analyser) {
        this.analyser = analyser;
        this.dataArray = new Uint8Array(analyser.frequencyBinCount);
        this.drawWaveform();
    }

    stopVisualization() {
        if (this.animationId) {
            cancelAnimationFrame(this.animationId);
            this.animationId = null;
        }

        // Clear canvas
        this.ctx.fillStyle = 'rgba(255, 255, 255, 0.05)';
        this.ctx.fillRect(0, 0, this.canvas.width, this.canvas.height);
    }

    drawWaveform() {
        if (!this.analyser) return;

        this.animationId = requestAnimationFrame(() => this.drawWaveform());

        // Get waveform data
        this.analyser.getByteTimeDomainData(this.dataArray);

        const width = this.canvas.width;
        const height = this.canvas.height;

        // Clear canvas with fade effect
        this.ctx.fillStyle = 'rgba(26, 26, 46, 0.2)';
        this.ctx.fillRect(0, 0, width, height);

        // Draw waveform with smoother line
        this.ctx.lineWidth = 2;
        this.ctx.strokeStyle = '#64ffda';
        this.ctx.lineCap = 'round';
        this.ctx.lineJoin = 'round';
        this.ctx.beginPath();

        // Sample fewer points for smoother rendering
        const step = Math.ceil(this.dataArray.length / 128);
        const sliceWidth = width / (this.dataArray.length / step);
        let x = 0;

        for (let i = 0; i < this.dataArray.length; i += step) {
            const v = this.dataArray[i] / 128.0;
            const y = v * height / 2;

            if (i === 0) {
                this.ctx.moveTo(x, y);
            } else {
                this.ctx.lineTo(x, y);
            }

            x += sliceWidth;
        }

        this.ctx.lineTo(width, height / 2);
        this.ctx.stroke();

        // Draw center line
        this.ctx.strokeStyle = 'rgba(100, 255, 218, 0.15)';
        this.ctx.lineWidth = 1;
        this.ctx.beginPath();
        this.ctx.moveTo(0, height / 2);
        this.ctx.lineTo(width, height / 2);
        this.ctx.stroke();
    }

    drawFrequencyBars() {
        if (!this.analyser) return;

        this.animationId = requestAnimationFrame(() => this.drawFrequencyBars());

        // Get frequency data
        this.analyser.getByteFrequencyData(this.dataArray);

        const width = this.canvas.width;
        const height = this.canvas.height;

        // Clear canvas
        this.ctx.fillStyle = 'rgba(26, 26, 46, 0.3)';
        this.ctx.fillRect(0, 0, width, height);

        // Draw bars
        const barCount = 64;
        const barWidth = width / barCount;
        const step = Math.floor(this.dataArray.length / barCount);

        for (let i = 0; i < barCount; i++) {
            const value = this.dataArray[i * step];
            const barHeight = (value / 255) * height * 0.8;

            const hue = (i / barCount) * 60 + 160;  // Cyan to green gradient
            this.ctx.fillStyle = `hsla(${hue}, 80%, 60%, 0.8)`;

            this.ctx.fillRect(
                i * barWidth,
                height - barHeight,
                barWidth - 1,
                barHeight
            );
        }
    }
}
