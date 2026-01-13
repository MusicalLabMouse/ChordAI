/**
 * Chord Display Module
 * Handles UI updates for chord detection display
 */

export class ChordDisplay {
  constructor() {
    // DOM elements
    this.statusDot = document.getElementById('statusDot');
    this.statusText = document.getElementById('statusText');
    this.chordDisplay = document.getElementById('chordDisplay');
    this.confidenceValue = document.getElementById('confidenceValue');
    this.errorMessage = document.getElementById('errorMessage');
    this.startBtn = document.getElementById('startBtn');
    this.stopBtn = document.getElementById('stopBtn');

    // Current state
    this.currentChord = '-';
    this.currentConfidence = 0;
  }

  /**
   * Set status indicator
   * @param {string} status - 'loading', 'ready', 'listening', 'error'
   * @param {string} message - Status message to display
   */
  setStatus(status, message) {
    // Remove all status classes
    this.statusDot.classList.remove('loading', 'ready', 'listening', 'error');

    // Add new status class
    this.statusDot.classList.add(status);

    // Update status text
    this.statusText.textContent = message;
  }

  /**
   * Update the chord display
   * @param {string} chord - Chord name to display
   * @param {number} confidence - Confidence value (0-1)
   */
  updateChord(chord, confidence) {
    // Format chord for display
    const displayChord = this.formatChordDisplay(chord);

    // Only update if chord changed
    if (displayChord !== this.currentChord) {
      this.currentChord = displayChord;
      this.chordDisplay.textContent = displayChord;

      // Update styling based on chord
      if (displayChord === '-' || displayChord === 'N') {
        this.chordDisplay.classList.add('no-chord');
      } else {
        this.chordDisplay.classList.remove('no-chord');
      }

      // Add a subtle animation on chord change
      this.chordDisplay.style.transform = 'scale(1.05)';
      setTimeout(() => {
        this.chordDisplay.style.transform = 'scale(1)';
      }, 100);
    }

    // Update confidence
    this.currentConfidence = confidence;
    const confidencePercent = Math.round(confidence * 100);
    this.confidenceValue.textContent = `${confidencePercent}%`;

    // Adjust opacity based on confidence
    const opacity = 0.5 + (confidence * 0.5);
    this.chordDisplay.style.opacity = opacity;
  }

  /**
   * Format chord for cleaner display
   */
  formatChordDisplay(chord) {
    if (!chord || chord === 'N') {
      return '-';
    }

    // Remove colon notation for cleaner look
    let display = chord.replace(':', '');

    // Keep it simple - just show the chord name
    return display;
  }

  /**
   * Show error message
   * @param {string} message - Error message
   */
  showError(message) {
    this.errorMessage.textContent = message;
    this.errorMessage.classList.add('visible');
    this.setStatus('error', 'Error');
  }

  /**
   * Hide error message
   */
  hideError() {
    this.errorMessage.classList.remove('visible');
  }

  /**
   * Enable start button
   */
  enableStart() {
    this.startBtn.disabled = false;
  }

  /**
   * Disable start button
   */
  disableStart() {
    this.startBtn.disabled = true;
  }

  /**
   * Enable stop button
   */
  enableStop() {
    this.stopBtn.disabled = false;
  }

  /**
   * Disable stop button
   */
  disableStop() {
    this.stopBtn.disabled = true;
  }

  /**
   * Set UI to listening state
   */
  setListening() {
    this.setStatus('listening', 'Listening...');
    this.disableStart();
    this.enableStop();
    this.hideError();
  }

  /**
   * Set UI to ready state
   */
  setReady() {
    this.setStatus('ready', 'Ready');
    this.enableStart();
    this.disableStop();
    this.updateChord('-', 0);
  }

  /**
   * Set UI to loading state
   */
  setLoading(message = 'Loading model...') {
    this.setStatus('loading', message);
    this.disableStart();
    this.disableStop();
  }

  /**
   * Reset display to initial state
   */
  reset() {
    this.updateChord('-', 0);
    this.hideError();
  }
}
