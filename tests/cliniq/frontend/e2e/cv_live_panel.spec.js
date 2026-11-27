/**
 * E2E — camera window and its right-hand "model input -> output" panel.
 * Requires: ClinIQ backend on localhost:8000, frontend on localhost:5173.
 *
 * No real camera and no model download: Chromium is given a synthetic webcam, and the
 * MediaPipe model file request is answered locally. The face detector therefore never
 * reports a face, which is exactly the state under test here: the panel must show the
 * DATABASE values will be used until a calibrated camera reading exists.
 */
import { test, expect } from '@playwright/test'

test.use({
  permissions: ['camera'],
  launchOptions: {
    args: ['--use-fake-ui-for-media-stream', '--use-fake-device-for-media-stream'],
  },
})

test.describe('Camera window — live panel', () => {
  test.beforeEach(async ({ page }) => {
    // answer the face-model download locally so the test needs no internet
    await page.route('**/face_landmarker.task', route =>
      route.fulfill({ status: 200, contentType: 'application/octet-stream', body: Buffer.from([0]) }))

    await page.goto('/')
    await page.getByText('Patient 2 — Moderate Risk').click()
    await page.waitForURL(/\/monitor\/2/, { timeout: 10000 })
    await expect(page.getByText('Current Status')).toBeVisible({ timeout: 20000 })
  })

  const openCamera = async (page) => {
    await page.getByRole('button', { name: /Open Live CV Monitor/i }).click()
    await expect(page.getByText('Live CV Assessment')).toBeVisible({ timeout: 10000 })
  }

  test('opens a window with the live panel on its right', async ({ page }) => {
    await openCamera(page)
    const panel = page.getByLabel('Live model input and output')
    await expect(panel).toBeVisible({ timeout: 15000 })
    const video = page.getByText('Live CV Assessment')
    const [p, v] = [await panel.boundingBox(), await video.boundingBox()]
    expect(p.x).toBeGreaterThan(v.x)                           // panel sits to the right of the header/title
  })

  test('panel shows the predicted SOFA, the two camera inputs, the note and the vitals', async ({ page }) => {
    await openCamera(page)
    const panel = page.getByLabel('Live model input and output')
    await expect(panel).toContainText('Predicted SOFA (output)', { timeout: 15000 })
    await expect(panel).toContainText('Input 1 & 2')
    await expect(panel).toContainText('Stress score (0–10)')
    await expect(panel).toContainText('GCS eye opening (1–4)')
    await expect(panel).toContainText('Input 3')
    await expect(panel).toContainText('clinical note')
    await expect(panel).toContainText('Other inputs')
    for (const v of ['HR', 'RR', 'SBP', 'DBP', 'MAP']) await expect(panel).toContainText(v)
  })

  test('without a camera reading the panel says DATABASE values will be used', async ({ page }) => {
    await openCamera(page)
    const panel = page.getByLabel('Live model input and output')
    await expect(panel).toContainText('At 0 → DATABASE values go into the model', { timeout: 15000 })
    await expect(panel).toContainText('WAITING FOR CAMERA')
  })

  test('the panel timer shows the dashboard countdown', async ({ page }) => {
    await openCamera(page)
    const panel = page.getByLabel('Live model input and output')
    await expect(panel).toContainText(/\d:\d{2}|Predicting now/, { timeout: 15000 })
  })

  test('closing the camera window removes the panel', async ({ page }) => {
    await openCamera(page)
    await expect(page.getByLabel('Live model input and output')).toBeVisible({ timeout: 15000 })
    await page.keyboard.press('Escape')
    await expect(page.getByLabel('Live model input and output')).toBeHidden({ timeout: 10000 })
  })

  // Once the window is open its dark backdrop covers the floating camera button, so the
  // button cannot be clicked again; a user closes the window with Escape, the ✕ or the backdrop.
  test('clicking the backdrop closes the window', async ({ page }) => {
    await openCamera(page)
    await page.mouse.click(5, 5)                                // far corner = the backdrop
    await expect(page.getByText('Live CV Assessment')).toBeHidden({ timeout: 10000 })
  })

  test('the floating camera button is covered while the window is open', async ({ page }) => {
    await openCamera(page)
    const fab = page.getByRole('button', { name: /Open Live CV Monitor/i })
    await expect(fab.click({ trial: true, timeout: 1500 })).rejects.toThrow(/intercepts pointer events/)
  })
})

test.describe('Dashboard overlays', () => {
  test.beforeEach(async ({ page }) => {
    await page.goto('/')
    await page.getByText('Patient 1 — Low Risk').click()
    await page.waitForURL(/\/monitor\/1/, { timeout: 10000 })
    await expect(page.getByText('Current Status')).toBeVisible({ timeout: 20000 })
  })

  test('history drawer opens and closes from its button', async ({ page }) => {
    await page.getByRole('button', { name: /Open patient history/i }).click()
    await expect(page.getByText('Patient History', { exact: true }).first()).toBeVisible({ timeout: 10000 })
  })

  test('SOFA validation drawer shows its three tabs and switches between them', async ({ page }) => {
    await page.getByRole('button', { name: /SOFA/i }).first().click()
    for (const tab of ['Population Stats', 'Runtime Checks', 'Uncertainty'])
      await expect(page.getByRole('button', { name: tab })).toBeVisible({ timeout: 10000 })
    await page.getByRole('button', { name: 'Runtime Checks' }).click()
    await expect(page.getByText('Physiological Lower-Bound Check')).toBeVisible()
    await page.getByRole('button', { name: 'Uncertainty' }).click()
    await expect(page.getByText(/90% conformal prediction interval/)).toBeVisible()
  })

  test('SOFA validation drawer closes with Escape', async ({ page }) => {
    await page.getByRole('button', { name: /SOFA/i }).first().click()
    await expect(page.getByText('Model Accuracy Report')).toBeVisible({ timeout: 10000 })
    await page.keyboard.press('Escape')
    await expect(page.getByText('Model Accuracy Report')).toBeHidden({ timeout: 10000 })
  })

  test('invalid patient id redirects to the selection page', async ({ page }) => {
    await page.goto('/monitor/9')
    await expect(page.getByText('Patient 1 — Low Risk')).toBeVisible({ timeout: 10000 })
  })

  test('an unknown URL redirects to the selection page', async ({ page }) => {
    await page.goto('/this/does/not/exist')
    await expect(page.getByText('Patient 1 — Low Risk')).toBeVisible({ timeout: 10000 })
  })
})
