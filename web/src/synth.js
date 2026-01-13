/**
 * Chord Synth Module
 * Plays detected chords with a warm, mellow tone and reverb
 */

export class ChordSynth {
  constructor() {
    this.audioContext = null;
    this.masterGain = null;
    this.reverbGain = null;
    this.dryGain = null;
    this.convolver = null;
    this.activeOscillators = [];
    this.isEnabled = false;
    this.volume = 0.5;
    this.currentChord = null;

    // Note frequencies (A4 = 440Hz)
    this.noteFrequencies = {
      'C': 261.63,
      'C#': 277.18,
      'D': 293.66,
      'D#': 311.13,
      'E': 329.63,
      'F': 349.23,
      'F#': 369.99,
      'G': 392.00,
      'G#': 415.30,
      'A': 440.00,
      'A#': 466.16,
      'B': 493.88
    };

    // Chord intervals (semitones from root)
    this.chordIntervals = {
      'maj': [0, 4, 7],
      'min': [0, 3, 7],
      'dim': [0, 3, 6],
      'aug': [0, 4, 8],
      'sus2': [0, 2, 7],
      'sus4': [0, 5, 7],
      'maj7': [0, 4, 7, 11],
      'min7': [0, 3, 7, 10],
      '7': [0, 4, 7, 10]
    };
  }

  /**
   * Initialize the audio context and effects chain
   */
  async init() {
    this.audioContext = new (window.AudioContext || window.webkitAudioContext)();

    // Master gain (volume control)
    this.masterGain = this.audioContext.createGain();
    this.masterGain.gain.value = this.volume;

    // Dry/wet mix for reverb
    this.dryGain = this.audioContext.createGain();
    this.dryGain.gain.value = 0.6;

    this.reverbGain = this.audioContext.createGain();
    this.reverbGain.gain.value = 0.4;

    // Create reverb using convolver
    this.convolver = this.audioContext.createConvolver();
    await this.createReverbImpulse();

    // Connect: source -> masterGain -> dry/wet split -> destination
    this.masterGain.connect(this.dryGain);
    this.masterGain.connect(this.convolver);
    this.convolver.connect(this.reverbGain);
    this.dryGain.connect(this.audioContext.destination);
    this.reverbGain.connect(this.audioContext.destination);

    console.log('ChordSynth initialized');
  }

  /**
   * Create a reverb impulse response
   */
  async createReverbImpulse() {
    const sampleRate = this.audioContext.sampleRate;
    const length = sampleRate * 2; // 2 second reverb
    const impulse = this.audioContext.createBuffer(2, length, sampleRate);

    for (let channel = 0; channel < 2; channel++) {
      const channelData = impulse.getChannelData(channel);
      for (let i = 0; i < length; i++) {
        // Exponential decay with some randomness for natural sound
        channelData[i] = (Math.random() * 2 - 1) * Math.pow(1 - i / length, 2.5);
      }
    }

    this.convolver.buffer = impulse;
  }

  /**
   * Parse chord string and get note frequencies
   */
  getChordFrequencies(chordStr) {
    if (!chordStr || chordStr === 'N' || chordStr === '-') {
      return [];
    }

    // Parse chord: "C:maj", "F#:min7", etc.
    let root, quality;
    if (chordStr.includes(':')) {
      [root, quality] = chordStr.split(':');
    } else {
      root = chordStr;
      quality = 'maj';
    }

    // Handle extensions in parentheses like "maj(b7)"
    quality = quality.replace(/\(.*\)/, '');

    // Get root frequency
    const rootFreq = this.noteFrequencies[root];
    if (!rootFreq) {
      console.warn(`Unknown root note: ${root}`);
      return [];
    }

    // Get intervals for this chord quality
    let intervals = this.chordIntervals[quality];
    if (!intervals) {
      // Default to major triad for unknown qualities
      intervals = this.chordIntervals['maj'];
    }

    // Calculate frequencies for each note in the chord
    // Use octave 3 (one below middle C) for warmth
    const baseFreq = rootFreq / 2;
    const frequencies = intervals.map(semitones => {
      return baseFreq * Math.pow(2, semitones / 12);
    });

    return frequencies;
  }

  /**
   * Create a warm oscillator with filtering
   */
  createWarmOscillator(frequency) {
    const osc1 = this.audioContext.createOscillator();
    const osc2 = this.audioContext.createOscillator();
    const filter = this.audioContext.createBiquadFilter();
    const oscGain = this.audioContext.createGain();

    // Main oscillator - sine wave for warmth
    osc1.type = 'sine';
    osc1.frequency.value = frequency;

    // Second oscillator - triangle, slightly detuned for richness
    osc2.type = 'triangle';
    osc2.frequency.value = frequency * 1.002; // Slight detune

    // Low-pass filter for mellow tone
    filter.type = 'lowpass';
    filter.frequency.value = 800;
    filter.Q.value = 0.5;

    // Gain for this voice
    oscGain.gain.value = 0.15;

    // Connect oscillators through filter
    osc1.connect(filter);
    osc2.connect(filter);
    filter.connect(oscGain);
    oscGain.connect(this.masterGain);

    return { osc1, osc2, filter, oscGain };
  }

  /**
   * Play a chord
   */
  playChord(chordStr) {
    if (!this.isEnabled || !this.audioContext) {
      return;
    }

    // Don't replay the same chord
    if (chordStr === this.currentChord) {
      return;
    }

    // Resume audio context if suspended (browser autoplay policy)
    if (this.audioContext.state === 'suspended') {
      this.audioContext.resume();
    }

    // Stop current chord
    this.stopChord();

    // Don't play silence
    if (!chordStr || chordStr === 'N' || chordStr === '-') {
      this.currentChord = chordStr;
      return;
    }

    const frequencies = this.getChordFrequencies(chordStr);
    if (frequencies.length === 0) {
      return;
    }

    this.currentChord = chordStr;
    const now = this.audioContext.currentTime;

    // Create oscillators for each note
    frequencies.forEach(freq => {
      const voice = this.createWarmOscillator(freq);

      // Smooth attack
      voice.oscGain.gain.setValueAtTime(0, now);
      voice.oscGain.gain.linearRampToValueAtTime(0.15, now + 0.1);

      voice.osc1.start(now);
      voice.osc2.start(now);

      this.activeOscillators.push(voice);
    });
  }

  /**
   * Stop current chord with smooth release
   */
  stopChord() {
    const now = this.audioContext?.currentTime || 0;

    this.activeOscillators.forEach(voice => {
      // Smooth release
      voice.oscGain.gain.linearRampToValueAtTime(0, now + 0.15);

      // Stop oscillators after release
      voice.osc1.stop(now + 0.2);
      voice.osc2.stop(now + 0.2);
    });

    this.activeOscillators = [];
  }

  /**
   * Set synth enabled/disabled
   */
  setEnabled(enabled) {
    this.isEnabled = enabled;
    if (!enabled) {
      this.stopChord();
      this.currentChord = null;
    }
    console.log(`Synth ${enabled ? 'enabled' : 'disabled'}`);
  }

  /**
   * Set volume (0-1)
   */
  setVolume(volume) {
    this.volume = Math.max(0, Math.min(1, volume));
    if (this.masterGain) {
      this.masterGain.gain.value = this.volume;
    }
  }

  /**
   * Toggle synth on/off
   */
  toggle() {
    this.setEnabled(!this.isEnabled);
    return this.isEnabled;
  }
}
