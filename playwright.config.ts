import { defineConfig, devices } from '@playwright/test';

/**
 * The e2e run never touches the model weights. It drives the app in `?demo=1` mode, where a
 * recorded run is replayed through the same worker protocol the real worker speaks — so the
 * test still exercises every view, every linked selection and the whole message boundary,
 * and CI stays fast and offline.
 */
export default defineConfig({
  testDir: './e2e',
  fullyParallel: true,
  forbidOnly: !!process.env.CI,
  retries: process.env.CI ? 1 : 0,
  workers: process.env.CI ? 1 : '50%',
  reporter: process.env.CI ? [['github'], ['html', { open: 'never' }]] : [['list']],
  use: {
    baseURL: 'http://127.0.0.1:4173/inside-the-llm/',
    trace: 'on-first-retry',
    screenshot: 'only-on-failure',
  },
  projects: [{ name: 'chromium', use: { ...devices['Desktop Chrome'] } }],
  webServer: {
    // --host 127.0.0.1: vite preview binds to `localhost` by default, which resolves to
    // ::1 on macOS and leaves the IPv4 address Playwright polls unanswered.
    command: 'npm run build && npx vite preview --host 127.0.0.1 --port 4173 --strictPort',
    url: 'http://127.0.0.1:4173/inside-the-llm/',
    reuseExistingServer: !process.env.CI,
    timeout: 180_000,
  },
});
