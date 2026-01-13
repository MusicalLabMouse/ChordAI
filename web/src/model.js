/**
 * ChordFormer Model Module
 * Handles ONNX Runtime inference and chord label reconstruction
 */

import * as ort from 'onnxruntime-web';

export class ChordModel {
  constructor(config) {
    this.config = config;
    this.session = null;
    this.isLoaded = false;

    // Chord mappings from config
    this.roots = config.mappings.roots;
    this.triads = config.mappings.triads;
    this.sevenths = config.mappings.sevenths;
    this.ninths = config.mappings.ninths;
    this.elevenths = config.mappings.elevenths;
    this.thirteenths = config.mappings.thirteenths;
  }

  /**
   * Load the ONNX model
   * @param {string} modelPath - Path to the .onnx file
   */
  async load(modelPath) {
    try {
      console.log('Loading ChordFormer model...');

      // Configure ONNX Runtime for performance
      ort.env.wasm.numThreads = navigator.hardwareConcurrency || 4; // Use available CPU cores
      ort.env.wasm.simd = true; // Enable SIMD acceleration

      // Create inference session
      this.session = await ort.InferenceSession.create(modelPath, {
        executionProviders: ['wasm'],
        graphOptimizationLevel: 'all'
      });

      this.isLoaded = true;
      console.log('Model loaded successfully');
      console.log('Input names:', this.session.inputNames);
      console.log('Output names:', this.session.outputNames);

    } catch (error) {
      console.error('Failed to load model:', error);
      throw error;
    }
  }

  /**
   * Run inference on features
   * @param {Float32Array[]} frames - Array of feature frames, each [nBins]
   * @returns {Object} - Predicted chord and confidence
   */
  async predict(frames) {
    if (!this.isLoaded) {
      throw new Error('Model not loaded');
    }

    const numFrames = frames.length;
    const nBins = this.config.nBins;

    // Create input tensor [1, numFrames, nBins]
    const inputData = new Float32Array(numFrames * nBins);
    for (let t = 0; t < numFrames; t++) {
      for (let b = 0; b < nBins; b++) {
        inputData[t * nBins + b] = frames[t][b];
      }
    }

    const inputTensor = new ort.Tensor('float32', inputData, [1, numFrames, nBins]);

    // Run inference
    const feeds = { 'features': inputTensor };
    const results = await this.session.run(feeds);

    // Get the last frame predictions (most recent)
    const lastFrameIdx = numFrames - 1;

    // Extract predictions for each head
    const rootTriadLogits = this.getFrameLogits(results['root_triad'], lastFrameIdx);
    const bassLogits = this.getFrameLogits(results['bass'], lastFrameIdx);
    const seventhLogits = this.getFrameLogits(results['7th'], lastFrameIdx);
    const ninthLogits = this.getFrameLogits(results['9th'], lastFrameIdx);
    const eleventhLogits = this.getFrameLogits(results['11th'], lastFrameIdx);
    const thirteenthLogits = this.getFrameLogits(results['13th'], lastFrameIdx);

    // Get argmax for each head
    const rootTriadIdx = this.argmax(rootTriadLogits);
    const bassIdx = this.argmax(bassLogits);
    const seventhIdx = this.argmax(seventhLogits);
    const ninthIdx = this.argmax(ninthLogits);
    const eleventhIdx = this.argmax(eleventhLogits);
    const thirteenthIdx = this.argmax(thirteenthLogits);

    // Calculate confidence (softmax probability of predicted class)
    const confidence = this.softmaxConfidence(rootTriadLogits, rootTriadIdx);

    // Reconstruct chord label
    const chord = this.reconstructChord(
      rootTriadIdx, bassIdx, seventhIdx, ninthIdx, eleventhIdx, thirteenthIdx
    );

    return {
      chord,
      confidence,
      rootTriadIdx,
      bassIdx,
      seventhIdx,
      ninthIdx,
      eleventhIdx,
      thirteenthIdx
    };
  }

  /**
   * Extract logits for a specific frame from output tensor
   */
  getFrameLogits(tensor, frameIdx) {
    const dims = tensor.dims; // [1, numFrames, numClasses]
    const numClasses = dims[2];
    const offset = frameIdx * numClasses;
    const data = tensor.data;

    const logits = new Float32Array(numClasses);
    for (let i = 0; i < numClasses; i++) {
      logits[i] = data[offset + i];
    }
    return logits;
  }

  /**
   * Find index of maximum value
   */
  argmax(arr) {
    let maxIdx = 0;
    let maxVal = arr[0];
    for (let i = 1; i < arr.length; i++) {
      if (arr[i] > maxVal) {
        maxVal = arr[i];
        maxIdx = i;
      }
    }
    return maxIdx;
  }

  /**
   * Calculate softmax confidence for the predicted class
   */
  softmaxConfidence(logits, predictedIdx) {
    // Stable softmax
    const maxLogit = Math.max(...logits);
    let sumExp = 0;
    for (let i = 0; i < logits.length; i++) {
      sumExp += Math.exp(logits[i] - maxLogit);
    }
    const confidence = Math.exp(logits[predictedIdx] - maxLogit) / sumExp;
    return confidence;
  }

  /**
   * Reconstruct chord label from structured predictions
   * Based on inference.py reconstruct_chord_label function
   */
  reconstructChord(rootTriadIdx, bassIdx, seventhIdx, ninthIdx, eleventhIdx, thirteenthIdx) {
    // No chord
    if (rootTriadIdx === 0) {
      return 'N';
    }

    // Decode root and triad
    // Index encoding: 1 + root * 7 + triad_type
    const adjustedIdx = rootTriadIdx - 1;
    const rootIdx = Math.floor(adjustedIdx / 7);
    const triadIdx = adjustedIdx % 7;

    const root = this.roots[rootIdx];

    // Triad type (0-5 = maj, min, sus4, sus2, dim, aug; 6 = N-quality)
    let triad = '';
    if (triadIdx < this.triads.length) {
      triad = this.triads[triadIdx];
    }

    // Build base chord label
    let chord;
    if (triad === 'maj') {
      chord = `${root}:maj`;
    } else if (triad === 'min') {
      chord = `${root}:min`;
    } else if (triad) {
      chord = `${root}:${triad}`;
    } else {
      chord = root;
    }

    // Collect extensions
    const extensions = [];

    if (seventhIdx > 0 && seventhIdx < this.sevenths.length) {
      const ext = this.sevenths[seventhIdx];
      if (ext !== 'N') {
        extensions.push(ext);
      }
    }

    if (ninthIdx > 0 && ninthIdx < this.ninths.length) {
      const ext = this.ninths[ninthIdx];
      if (ext !== 'N') {
        extensions.push(ext);
      }
    }

    if (eleventhIdx > 0 && eleventhIdx < this.elevenths.length) {
      const ext = this.elevenths[eleventhIdx];
      if (ext !== 'N') {
        extensions.push(ext);
      }
    }

    if (thirteenthIdx > 0 && thirteenthIdx < this.thirteenths.length) {
      const ext = this.thirteenths[thirteenthIdx];
      if (ext !== 'N') {
        extensions.push(ext);
      }
    }

    // Add extensions to chord label
    if (extensions.length > 0) {
      const extStr = extensions.join('');

      // Simplify common patterns
      if (chord.includes(':maj') && extStr.startsWith('7')) {
        chord = chord.replace(':maj', ':maj7');
        if (extStr.length > 1) {
          chord += `(${extStr.substring(1)})`;
        }
      } else if (chord.includes(':min') && extStr.startsWith('b7')) {
        chord = chord.replace(':min', ':min7');
        if (extStr.length > 2) {
          chord += `(${extStr.substring(2)})`;
        }
      } else {
        chord += `(${extStr})`;
      }
    }

    // Add bass note if different from root
    if (bassIdx > 0 && bassIdx <= this.roots.length) {
      const bassNote = this.roots[bassIdx - 1];
      if (bassNote !== root) {
        chord += `/${bassNote}`;
      }
    }

    return chord;
  }

  /**
   * Get a simplified display version of the chord
   */
  getDisplayChord(chord) {
    if (chord === 'N') {
      return '-';
    }

    // Remove colon for cleaner display
    let display = chord.replace(':', '');

    // Simplify maj to just the root for basic major chords
    if (display.endsWith('maj') && !display.includes('7')) {
      display = display.replace('maj', '');
    }

    return display;
  }

  /**
   * Remove slash chord notation (e.g., "C:maj/E" -> "C:maj")
   */
  stripSlashChord(chord) {
    if (chord.includes('/')) {
      return chord.split('/')[0];
    }
    return chord;
  }

  /**
   * Check if model is loaded
   */
  isReady() {
    return this.isLoaded;
  }
}
