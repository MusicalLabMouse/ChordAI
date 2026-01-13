/**
 * CQT Feature Extraction Module
 * Extracts Constant-Q Transform features from audio for chord recognition
 */

export class FeatureExtractor {
  constructor(config, normalization) {
    this.sampleRate = config.sampleRate || 22050;
    this.hopLength = config.hopLength || 512;
    this.nBins = config.nBins || 252;
    this.binsPerOctave = config.binsPerOctave || 36;
    this.fmin = config.fmin || 32.7; // C1

    // Normalization parameters
    this.mean = normalization?.mean || null;
    this.std = normalization?.std || null;

    // FFT size (reduced for lower latency, trades off some bass resolution)
    this.fftSize = 4096;

    // Pre-compute CQT filterbank
    this.filterbank = this.createCQTFilterbank();

    // Pre-compute Hann window
    this.hannWindow = this.createHannWindow(this.fftSize);

    // Pre-compute FFT twiddle factors
    this.fft = new FFT(this.fftSize);
  }

  /**
   * Create Hann window for FFT
   */
  createHannWindow(size) {
    const window = new Float32Array(size);
    for (let i = 0; i < size; i++) {
      window[i] = 0.5 * (1 - Math.cos(2 * Math.PI * i / (size - 1)));
    }
    return window;
  }

  /**
   * Create CQT filterbank mapping FFT bins to CQT bins
   * Uses triangular weighting for smooth interpolation
   */
  createCQTFilterbank() {
    const filterbank = [];

    // Calculate center frequencies for each CQT bin
    for (let k = 0; k < this.nBins; k++) {
      // Center frequency for this CQT bin
      const centerFreq = this.fmin * Math.pow(2, k / this.binsPerOctave);

      // Q factor (constant for CQT)
      const Q = 1.0 / (Math.pow(2, 1.0 / this.binsPerOctave) - 1);

      // Bandwidth
      const bandwidth = centerFreq / Q;

      // Lower and upper frequencies
      const fLow = centerFreq - bandwidth / 2;
      const fHigh = centerFreq + bandwidth / 2;

      // Convert to FFT bin indices
      const binLow = Math.max(0, Math.floor(fLow * this.fftSize / this.sampleRate));
      const binCenter = Math.round(centerFreq * this.fftSize / this.sampleRate);
      const binHigh = Math.min(this.fftSize / 2, Math.ceil(fHigh * this.fftSize / this.sampleRate));

      // Create triangular filter weights
      const filter = {
        binStart: binLow,
        binEnd: binHigh,
        weights: []
      };

      for (let bin = binLow; bin <= binHigh; bin++) {
        let weight;
        if (bin <= binCenter) {
          // Rising edge
          weight = (bin - binLow) / Math.max(1, binCenter - binLow);
        } else {
          // Falling edge
          weight = (binHigh - bin) / Math.max(1, binHigh - binCenter);
        }
        filter.weights.push(weight);
      }

      filterbank.push(filter);
    }

    return filterbank;
  }

  /**
   * Extract CQT features from audio samples
   * @param {Float32Array} samples - Audio samples (should be fftSize = 8192)
   * @returns {Float32Array} - CQT features (nBins dimensions)
   */
  extractFrame(samples) {
    // Use samples directly (should already be fftSize)
    const windowedSamples = new Float32Array(this.fftSize);

    // Copy and apply Hann window
    const copyLength = Math.min(samples.length, this.fftSize);
    for (let i = 0; i < copyLength; i++) {
      windowedSamples[i] = samples[i] * this.hannWindow[i];
    }

    // Compute FFT
    const spectrum = this.fft.forward(windowedSamples);

    // Compute magnitude spectrum with normalization
    // Divide by fftSize to normalize FFT output to match librosa's scale
    const magnitudes = new Float32Array(this.fftSize / 2 + 1);
    const normFactor = this.fftSize;
    for (let i = 0; i <= this.fftSize / 2; i++) {
      const re = spectrum.real[i];
      const im = spectrum.imag[i];
      magnitudes[i] = Math.sqrt(re * re + im * im) / normFactor;
    }

    // Apply CQT filterbank
    const cqt = new Float32Array(this.nBins);
    for (let k = 0; k < this.nBins; k++) {
      const filter = this.filterbank[k];
      let sum = 0;

      for (let j = 0; j < filter.weights.length; j++) {
        const binIndex = filter.binStart + j;
        if (binIndex < magnitudes.length) {
          sum += magnitudes[binIndex] * filter.weights[j];
        }
      }

      cqt[k] = sum;
    }

    // Convert to dB scale with max reference (matching librosa's ref=np.max)
    // This normalizes so the maximum value becomes 0 dB
    const cqtDb = new Float32Array(this.nBins);

    // Find max amplitude for reference (like librosa's ref=np.max)
    let maxAmplitude = 0;
    for (let k = 0; k < this.nBins; k++) {
      if (cqt[k] > maxAmplitude) {
        maxAmplitude = cqt[k];
      }
    }
    const ref = Math.max(maxAmplitude, 1e-10);  // Avoid division by zero

    for (let k = 0; k < this.nBins; k++) {
      // amplitude_to_db: 20 * log10(amplitude / ref)
      // With ref=max, the maximum becomes 0 dB, others are negative
      const amplitude = Math.max(cqt[k], 1e-10);
      cqtDb[k] = 20 * Math.log10(amplitude / ref);
      // Apply floor to match librosa's top_db=80 default
      cqtDb[k] = Math.max(cqtDb[k], -80);
    }

    return cqtDb;
  }

  /**
   * Normalize features using z-score normalization
   * @param {Float32Array} features - Raw CQT features
   * @returns {Float32Array} - Normalized features
   */
  normalize(features) {
    if (!this.mean || !this.std) {
      return features;
    }

    const normalized = new Float32Array(features.length);
    for (let i = 0; i < features.length; i++) {
      normalized[i] = (features[i] - this.mean[i]) / (this.std[i] || 1);
    }

    return normalized;
  }

  /**
   * Extract and normalize a single frame
   * @param {Float32Array} samples - Audio samples
   * @returns {Float32Array} - Normalized CQT features
   */
  processFrame(samples) {
    const features = this.extractFrame(samples);
    return this.normalize(features);
  }
}


/**
 * Simple FFT implementation using Cooley-Tukey algorithm
 */
class FFT {
  constructor(size) {
    this.size = size;
    this.log2Size = Math.log2(size);

    if (Math.pow(2, this.log2Size) !== size) {
      throw new Error('FFT size must be a power of 2');
    }

    // Pre-compute bit-reversal permutation
    this.reverseTable = new Uint32Array(size);
    for (let i = 0; i < size; i++) {
      this.reverseTable[i] = this.reverseBits(i, this.log2Size);
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

  forward(input) {
    const n = this.size;
    const real = new Float32Array(n);
    const imag = new Float32Array(n);

    // Bit-reversal permutation
    for (let i = 0; i < n; i++) {
      real[i] = input[this.reverseTable[i]];
      imag[i] = 0;
    }

    // Cooley-Tukey FFT
    for (let size = 2; size <= n; size *= 2) {
      const halfSize = size / 2;
      const step = n / size;

      for (let i = 0; i < n; i += size) {
        for (let j = 0; j < halfSize; j++) {
          const k = j * step;
          const cos = this.cosTable[k];
          const sin = this.sinTable[k];

          const evenIdx = i + j;
          const oddIdx = i + j + halfSize;

          const evenReal = real[evenIdx];
          const evenImag = imag[evenIdx];
          const oddReal = real[oddIdx];
          const oddImag = imag[oddIdx];

          // Butterfly operation
          const tReal = cos * oddReal - sin * oddImag;
          const tImag = sin * oddReal + cos * oddImag;

          real[evenIdx] = evenReal + tReal;
          imag[evenIdx] = evenImag + tImag;
          real[oddIdx] = evenReal - tReal;
          imag[oddIdx] = evenImag - tImag;
        }
      }
    }

    return { real, imag };
  }
}
