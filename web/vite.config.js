import { defineConfig } from 'vite';

export default defineConfig({
  // Optimize deps for ONNX Runtime
  optimizeDeps: {
    exclude: ['onnxruntime-web']
  },

  // Configure server for proper WASM/SharedArrayBuffer support
  server: {
    headers: {
      'Cross-Origin-Opener-Policy': 'same-origin',
      'Cross-Origin-Embedder-Policy': 'require-corp'
    }
  },

  // Build configuration
  build: {
    target: 'esnext',
    outDir: 'dist'
  },

  // Ensure WASM files are served correctly
  assetsInclude: ['**/*.onnx', '**/*.wasm']
});
