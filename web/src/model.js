/**
 * ONNX Model Inference Module
 * Handles loading and running the chord detection model using ONNX Runtime Web
 * Note: ort is loaded globally via script tag in index.html
 */

export class ChordModel {
    constructor() {
        this.session = null;
        this.chordMapping = null;
        this.modelConfig = null;
        this.isLoaded = false;
    }

    async load(modelPath = '/model/chord_model.onnx') {
        try {
            // Check if ort is available (loaded via script tag)
            if (typeof ort === 'undefined') {
                throw new Error('ONNX Runtime not loaded. Check script tag in index.html');
            }

            console.log('Fetching ONNX model...');

            // Fetch model as ArrayBuffer (more reliable than URL loading)
            const response = await fetch(modelPath);
            if (!response.ok) {
                throw new Error(`Failed to fetch model: ${response.status} ${response.statusText}`);
            }
            const modelBuffer = await response.arrayBuffer();
            console.log(`Model size: ${(modelBuffer.byteLength / 1024 / 1024).toFixed(2)} MB`);

            console.log('Creating inference session...');

            // Use WASM with SIMD (threading requires cross-origin isolation)
            ort.env.wasm.simd = true;

            const options = {
                executionProviders: ['wasm'],
                graphOptimizationLevel: 'all',
                enableCpuMemArena: true
            };

            // Load the model from ArrayBuffer
            this.session = await ort.InferenceSession.create(modelBuffer, options);
            console.log('Model loaded successfully');
            console.log('Input names:', this.session.inputNames);
            console.log('Output names:', this.session.outputNames);

            // Load chord mapping
            await this.loadChordMapping();

            // Load model config
            await this.loadModelConfig();

            this.isLoaded = true;

        } catch (error) {
            console.error('Failed to load model:', error);
            console.error('Error type:', typeof error);
            console.error('Error constructor:', error?.constructor?.name);
            if (error instanceof Error) {
                throw new Error(`Failed to load model: ${error.message}`);
            } else {
                throw new Error(`Failed to load model: ${JSON.stringify(error)}`);
            }
        }
    }

    async loadChordMapping() {
        try {
            const response = await fetch('/model/chord_mapping.json');
            if (!response.ok) {
                console.warn('Chord mapping not found, using defaults');
                this.setDefaultMapping();
                return;
            }

            const data = await response.json();
            this.chordMapping = data.idx_to_chord;
            console.log(`Loaded ${Object.keys(this.chordMapping).length} chord labels`);
        } catch (error) {
            console.warn('Failed to load chord mapping:', error);
            this.setDefaultMapping();
        }
    }

    setDefaultMapping() {
        // Default 25-class mapping
        this.chordMapping = {
            '0': 'N',
            '1': 'A', '2': 'Am',
            '3': 'Ab', '4': 'Abm',
            '5': 'B', '6': 'Bm',
            '7': 'Bb', '8': 'Bbm',
            '9': 'C', '10': 'Cm',
            '11': 'D', '12': 'Dm',
            '13': 'Db', '14': 'Dbm',
            '15': 'E', '16': 'Em',
            '17': 'Eb', '18': 'Ebm',
            '19': 'F#', '20': 'F#m',
            '21': 'F', '22': 'Fm',
            '23': 'G', '24': 'Gm'
        };
    }

    async loadModelConfig() {
        try {
            const response = await fetch('/model/model_config.json');
            if (!response.ok) {
                console.warn('Model config not found');
                return;
            }

            this.modelConfig = await response.json();
            console.log('Model config:', this.modelConfig);
        } catch (error) {
            console.warn('Failed to load model config:', error);
        }
    }

    async predict(features) {
        if (!this.isLoaded) {
            throw new Error('Model not loaded');
        }

        const { data, shape } = features;

        // Create input tensor
        const inputTensor = new ort.Tensor('float32', data, shape);

        // Run inference
        const feeds = { 'features': inputTensor };
        const results = await this.session.run(feeds);

        // Get output
        const output = results['predictions'];
        const outputData = output.data;
        const outputShape = output.dims;

        // Process predictions
        const predictions = this.processOutput(outputData, outputShape);

        return predictions;
    }

    processOutput(outputData, outputShape) {
        const [batchSize, numFrames, numClasses] = outputShape;
        const predictions = [];
        const confidences = [];

        // Process each frame
        for (let f = 0; f < numFrames; f++) {
            const offset = f * numClasses;
            const frameLogits = Array.from(outputData.slice(offset, offset + numClasses));

            // Softmax
            const maxLogit = Math.max(...frameLogits);
            const expLogits = frameLogits.map(x => Math.exp(x - maxLogit));
            const sumExp = expLogits.reduce((a, b) => a + b, 0);
            const probs = expLogits.map(x => x / sumExp);

            // Get predicted class
            const predictedIdx = probs.indexOf(Math.max(...probs));
            const confidence = probs[predictedIdx];

            // Map to chord name
            const chordName = this.formatChordName(this.chordMapping[predictedIdx.toString()] || 'N');

            predictions.push(chordName);
            confidences.push(confidence);
        }

        return { predictions, confidences };
    }

    formatChordName(chord) {
        if (chord === 'N') return '-';

        // Convert from model format to display format
        // e.g., "A:maj" -> "A", "A:min" -> "Am"
        if (chord.includes(':maj')) {
            return chord.replace(':maj', '');
        } else if (chord.includes(':min')) {
            return chord.replace(':min', 'm');
        }

        return chord;
    }

    getNumClasses() {
        return this.chordMapping ? Object.keys(this.chordMapping).length : 25;
    }
}
