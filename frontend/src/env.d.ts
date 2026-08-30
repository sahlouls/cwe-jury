/// <reference types="vite/client" />

// Augmentation de l'env Vite : URL de l'API (defaut gere dans lib/api.ts).
interface ImportMetaEnv {
  readonly VITE_API_URL?: string;
}

interface ImportMeta {
  readonly env: ImportMetaEnv;
}
