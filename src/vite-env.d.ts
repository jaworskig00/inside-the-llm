/// <reference types="vite/client" />

interface ImportMetaEnv {
  /** Where the exported `model.onnx` is hosted. Defaults to this repo's GitHub Release. */
  readonly VITE_MODEL_URL?: string;
}

interface ImportMeta {
  readonly env: ImportMetaEnv;
}
