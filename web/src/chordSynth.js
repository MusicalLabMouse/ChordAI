/**
 * Chord Synthesizer Module
 * Plays detected chords using Web Audio API
 */

export class ChordSynth {
    constructor() {
        this.audioContext = null;
        this.gainNode = null;
        this.activeOscillators = [];
        this.isEnabled = false;
        this.volume = 0.5;
        this.currentChord = null;

        // Note frequencies (C4 = middle C)
        this.noteFrequencies = {
            'C': 261.63,
            'Db': 277.18,
            'D': 293.66,
            'Eb': 311.13,
            'E': 329.63,
            'F': 349.23,
            'F#': 369.99,
            'G': 392.00,
            'Ab': 415.30,
            'A': 440.00,
            'Bb': 466.16,
            'B': 493.88
        };
    }

    init() {
        this.audioContext = new (window.AudioContext || window.webkitAudioContext)({
            latencyHint: 'interactive',
            sampleRate: 44100
        });

        // Create reverb
        this.convolver = this.audioContext.createConvolver();
        this.createReverb(2.5, 2);  // 2.5 second decay, mellow

        // Dry/wet mix
        this.dryGain = this.audioContext.createGain();
        this.wetGain = this.audioContext.createGain();
        this.dryGain.gain.value = 0.4;
        this.wetGain.gain.value = 0.6;

        // Master gain
        this.gainNode = this.audioContext.createGain();
        this.gainNode.gain.value = this.volume;

        // Routing: gainNode -> dry -> destination
        //          gainNode -> convolver -> wet -> destination
        this.gainNode.connect(this.dryGain);
        this.gainNode.connect(this.convolver);
        this.convolver.connect(this.wetGain);
        this.dryGain.connect(this.audioContext.destination);
        this.wetGain.connect(this.audioContext.destination);
    }

    createReverb(duration, decay) {
        const sampleRate = this.audioContext.sampleRate;
        const length = sampleRate * duration;
        const impulse = this.audioContext.createBuffer(2, length, sampleRate);

        for (let channel = 0; channel < 2; channel++) {
            const channelData = impulse.getChannelData(channel);
            for (let i = 0; i < length; i++) {
                channelData[i] = (Math.random() * 2 - 1) * Math.pow(1 - i / length, decay);
            }
        }

        this.convolver.buffer = impulse;
    }

    enable() {
        if (!this.audioContext) {
            this.init();
        }
        this.isEnabled = true;
        if (this.audioContext.state === 'suspended') {
            this.audioContext.resume();
        }
    }

    disable() {
        this.isEnabled = false;
        this.stopAllNotes();
    }

    toggle() {
        if (this.isEnabled) {
            this.disable();
        } else {
            this.enable();
        }
        return this.isEnabled;
    }

    setVolume(value) {
        this.volume = Math.max(0, Math.min(1, value));
        if (this.gainNode) {
            this.gainNode.gain.value = this.volume;
        }
    }

    /**
     * Parse chord name into root and quality
     */
    parseChord(chordName) {
        if (!chordName || chordName === '-' || chordName === 'N') {
            return null;
        }

        if (chordName.endsWith('m') && !chordName.endsWith('#m')) {
            return { root: chordName.slice(0, -1), quality: 'min' };
        }

        if (chordName.endsWith('#m')) {
            return { root: chordName.slice(0, -1), quality: 'min' };
        }

        return { root: chordName, quality: 'maj' };
    }

    /**
     * Get piano voicing frequencies for a chord
     */
    getChordFrequencies(chordName) {
        const parsed = this.parseChord(chordName);
        if (!parsed) return null;

        const { root, quality } = parsed;
        const rootFreq = this.noteFrequencies[root];
        if (!rootFreq) return null;

        const third = quality === 'min' ? 3 : 4;
        const fifth = 7;

        // Piano voicing spread across registers
        return [
            rootFreq / 2,                              // Bass note (octave below)
            rootFreq,                                   // Root
            rootFreq * Math.pow(2, third / 12),        // 3rd
            rootFreq * Math.pow(2, fifth / 12),        // 5th
            rootFreq * 2,                              // Root (octave above)
            rootFreq * Math.pow(2, (third + 12) / 12), // 3rd (octave above)
        ];
    }

    /**
     * Play a chord with mellow warm sound
     */
    playChord(chordName) {
        if (!this.isEnabled || !this.audioContext) return;

        if (chordName === this.currentChord) return;
        this.currentChord = chordName;

        this.stopAllNotes();

        const frequencies = this.getChordFrequencies(chordName);
        if (!frequencies) return;

        const now = this.audioContext.currentTime;

        frequencies.forEach(freq => {
            // Warm mellow filter
            const filter = this.audioContext.createBiquadFilter();
            filter.type = 'lowpass';
            filter.frequency.value = 600;
            filter.Q.value = 0.5;

            const noteGain = this.audioContext.createGain();
            noteGain.gain.value = 0;

            filter.connect(noteGain);
            noteGain.connect(this.gainNode);

            // Fundamental (sine for purity)
            const osc1 = this.audioContext.createOscillator();
            osc1.type = 'sine';
            osc1.frequency.value = freq;
            osc1.connect(filter);
            osc1.start(now);

            // Soft sub octave for warmth
            const osc2 = this.audioContext.createOscillator();
            osc2.type = 'sine';
            osc2.frequency.value = freq / 2;
            const subGain = this.audioContext.createGain();
            subGain.gain.value = 0.3;
            osc2.connect(subGain);
            subGain.connect(filter);
            osc2.start(now);

            // Gentle attack, long sustain
            noteGain.gain.setValueAtTime(0, now);
            noteGain.gain.linearRampToValueAtTime(0.2, now + 0.15);   // Slow attack
            noteGain.gain.linearRampToValueAtTime(0.18, now + 0.8);   // Gentle sustain

            this.activeOscillators.push(
                { osc: osc1, gain: noteGain },
                { osc: osc2, gain: noteGain }
            );
        });
    }

    stopAllNotes() {
        if (!this.audioContext) return;

        const now = this.audioContext.currentTime;

        this.activeOscillators.forEach(({ osc, gain }) => {
            try {
                gain.gain.linearRampToValueAtTime(0, now + 0.1);
                osc.stop(now + 0.15);
            } catch (e) {}
        });

        this.activeOscillators = [];
    }

    destroy() {
        this.stopAllNotes();
        if (this.audioContext) {
            this.audioContext.close();
            this.audioContext = null;
        }
    }
}
