/**
 * Feature Extraction Module
 * Implements CQT (Constant-Q Transform) approximation for browser-based audio analysis
 *
 * Note: This is a simplified CQT implementation using FFT.
 * For production, consider using Essentia.js for more accurate CQT extraction.
 */

export class FeatureExtractor {
    constructor(options = {}) {
        this.sampleRate = options.sampleRate || 22050;
        this.nBins = options.nBins || 84;  // 7 octaves * 12 bins
        this.binsPerOctave = options.binsPerOctave || 12;
        this.fmin = options.fmin || 32.7;  // C1

        // Normalization parameters (loaded from model config)
        this.mean = null;
        this.std = null;

        // Pre-compute CQT frequency bins
        this.frequencies = this.computeFrequencies();
        this.fftSize = 2048;  // Reduced for better performance

        // Pre-compute CQT kernel mapping from FFT bins to CQT bins
        this.kernelMapping = this.computeKernelMapping();

        // Initialize optimized FFT
        this.fft = new FFT(this.fftSize);
        this.fftReal = new Float32Array(this.fftSize);
        this.fftImag = new Float32Array(this.fftSize);
    }

    computeFrequencies() {
        // Compute center frequencies for each CQT bin
        const frequencies = new Float32Array(this.nBins);
        for (let i = 0; i < this.nBins; i++) {
            frequencies[i] = this.fmin * Math.pow(2, i / this.binsPerOctave);
        }
        return frequencies;
    }

    computeKernelMapping() {
        // Map FFT bins to CQT bins
        // Each CQT bin corresponds to a range of FFT bins
        const mapping = [];
        const fftBinWidth = this.sampleRate / this.fftSize;

        for (let i = 0; i < this.nBins; i++) {
            const centerFreq = this.frequencies[i];
            // Q factor for constant-Q
            const Q = 1 / (Math.pow(2, 1 / this.binsPerOctave) - 1);
            const bandwidth = centerFreq / Q;

            const lowFreq = centerFreq - bandwidth / 2;
            const highFreq = centerFreq + bandwidth / 2;

            const lowBin = Math.max(0, Math.floor(lowFreq / fftBinWidth));
            const highBin = Math.min(this.fftSize / 2, Math.ceil(highFreq / fftBinWidth));

            mapping.push({ lowBin, highBin, centerFreq });
        }

        return mapping;
    }

    async loadNormalization() {
        try {
            const response = await fetch('/model/normalization.json');
            if (!response.ok) {
                console.warn('Normalization file not found, using defaults');
                this.setDefaultNormalization();
                return;
            }

            const data = await response.json();
            this.mean = new Float32Array(data.mean);
            this.std = new Float32Array(data.std);
            console.log('Loaded normalization parameters');
        } catch (error) {
            console.warn('Failed to load normalization:', error);
            this.setDefaultNormalization();
        }
    }

    setDefaultNormalization() {
        // Default normalization (approximate values from training)
        this.mean = new Float32Array(this.nBins).fill(-3.5);
        this.std = new Float32Array(this.nBins).fill(1.8);
    }

    extractCQT(samples) {
        // Apply window function (Hann window)
        const windowed = this.applyWindow(samples);

        // Compute FFT
        const fftResult = this.computeFFT(windowed);

        // Convert FFT to CQT bins
        const cqt = this.fftToCQT(fftResult);

        // Apply log scaling
        const logCqt = this.logScale(cqt);

        // Normalize
        const normalized = this.normalize(logCqt);

        return normalized;
    }

    applyWindow(samples) {
        const n = samples.length;
        const windowed = new Float32Array(n);

        for (let i = 0; i < n; i++) {
            // Hann window
            const window = 0.5 * (1 - Math.cos(2 * Math.PI * i / (n - 1)));
            windowed[i] = samples[i] * window;
        }

        return windowed;
    }

    computeFFT(samples) {
        // Zero-pad/copy to FFT buffers
        this.fftReal.fill(0);
        this.fftImag.fill(0);
        const copyLen = Math.min(samples.length, this.fftSize);
        this.fftReal.set(samples.subarray(0, copyLen));

        // Use optimized Cooley-Tukey FFT (O(n log n) instead of O(n²))
        this.fft.forward(this.fftReal, this.fftImag);

        // Compute magnitude (only need first half due to symmetry)
        const magnitude = new Float32Array(this.fftSize / 2);
        for (let i = 0; i < this.fftSize / 2; i++) {
            magnitude[i] = Math.sqrt(
                this.fftReal[i] * this.fftReal[i] +
                this.fftImag[i] * this.fftImag[i]
            );
        }

        return magnitude;
    }

    fftToCQT(fftMagnitude) {
        const cqt = new Float32Array(this.nBins);

        for (let i = 0; i < this.nBins; i++) {
            const { lowBin, highBin } = this.kernelMapping[i];

            // Average magnitude over the frequency range
            let sum = 0;
            let count = 0;

            for (let j = lowBin; j <= highBin && j < fftMagnitude.length; j++) {
                sum += fftMagnitude[j];
                count++;
            }

            cqt[i] = count > 0 ? sum / count : 0;
        }

        return cqt;
    }

    logScale(cqt) {
        const logCqt = new Float32Array(cqt.length);
        const epsilon = 1e-6;

        for (let i = 0; i < cqt.length; i++) {
            logCqt[i] = Math.log(cqt[i] + epsilon);
        }

        return logCqt;
    }

    normalize(features) {
        if (!this.mean || !this.std) {
            return features;
        }

        const normalized = new Float32Array(features.length);

        for (let i = 0; i < features.length; i++) {
            normalized[i] = (features[i] - this.mean[i]) / this.std[i];
        }

        return normalized;
    }
}

/**
 * Optimized FFT using Cooley-Tukey algorithm
 * For better performance in production
 */
export class FFT {
    constructor(size) {
        this.size = size;
        this.levels = Math.log2(size);

        if (Math.pow(2, this.levels) !== size) {
            throw new Error('FFT size must be a power of 2');
        }

        // Pre-compute bit reversal indices
        this.bitReversal = new Uint32Array(size);
        for (let i = 0; i < size; i++) {
            this.bitReversal[i] = this.reverseBits(i, this.levels);
        }

        // Pre-compute twiddle factors
        this.cosTable = new Float32Array(size / 2);
        this.sinTable = new Float32Array(size / 2);
        for (let i = 0; i < size / 2; i++) {
            const angle = -2 * Math.PI * i / size;
            this.cosTable[i] = Math.cos(angle);
            this.sinTable[i] = Math.sin(angle);
        }
    }

    reverseBits(x, bits) {
        let result = 0;
        for (let i = 0; i < bits; i++) {
            result = (result << 1) | (x & 1);
            x >>= 1;
        }
        return result;
    }

    forward(real, imag) {
        const n = this.size;

        // Bit reversal permutation
        for (let i = 0; i < n; i++) {
            const j = this.bitReversal[i];
            if (j > i) {
                [real[i], real[j]] = [real[j], real[i]];
                [imag[i], imag[j]] = [imag[j], imag[i]];
            }
        }

        // Cooley-Tukey FFT
        for (let size = 2; size <= n; size *= 2) {
            const halfSize = size / 2;
            const tableStep = n / size;

            for (let i = 0; i < n; i += size) {
                for (let j = 0; j < halfSize; j++) {
                    const k = j * tableStep;
                    const tReal = real[i + j + halfSize] * this.cosTable[k] -
                                  imag[i + j + halfSize] * this.sinTable[k];
                    const tImag = real[i + j + halfSize] * this.sinTable[k] +
                                  imag[i + j + halfSize] * this.cosTable[k];

                    real[i + j + halfSize] = real[i + j] - tReal;
                    imag[i + j + halfSize] = imag[i + j] - tImag;
                    real[i + j] += tReal;
                    imag[i + j] += tImag;
                }
            }
        }
    }
}
