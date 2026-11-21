import { defineConfig, devices } from '@playwright/test'

// E2E specs live outside frontend/ with the rest of the ClinIQ tests.
// Requires the backend on :8000 (uvicorn backend.main:app --port 8000);
// the Vite dev server is started automatically if it isn't already running.
export default defineConfig({
  testDir: '../tests/cliniq/frontend/e2e',
  testMatch: '**/*.spec.js',
  timeout: 120_000,
  fullyParallel: false,
  workers: 1,
  reporter: [['list']],
  use: {
    baseURL: 'http://localhost:5173',
    trace: 'retain-on-failure',
  },
  projects: [{ name: 'chromium', use: { ...devices['Desktop Chrome'] } }],
  webServer: {
    command: 'npx vite --port 5173 --strictPort',
    url: 'http://localhost:5173',
    reuseExistingServer: true,
    timeout: 60_000,
  },
})
