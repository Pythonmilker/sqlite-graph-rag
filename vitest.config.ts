import { defineConfig } from 'vitest/config';
import { fileURLToPath } from 'node:url';
import { dirname, resolve } from 'node:path';

const here = dirname(fileURLToPath(import.meta.url));

export default defineConfig({
  resolve: {
    alias: {
      '@hub/core': resolve(here, '../core/src/index.ts'),
      '@hub/store': resolve(here, '../store/src/index.ts'),
    },
  },
});
