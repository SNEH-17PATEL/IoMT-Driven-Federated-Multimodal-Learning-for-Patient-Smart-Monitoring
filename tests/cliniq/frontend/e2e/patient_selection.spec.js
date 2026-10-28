/**
 * E2E — Patient selection flow
 * Requires: backend on localhost:8000, frontend on localhost:5173
 * Run: npx playwright test
 */
import { test, expect } from '@playwright/test'

test.describe('Patient Selection', () => {
  test('home page shows 3 patient cards', async ({ page }) => {
    await page.goto('/')
    await expect(page.getByText('Patient 1 — Low Risk')).toBeVisible({ timeout: 10000 })
    await expect(page.getByText('Patient 2 — Moderate Risk')).toBeVisible()
    await expect(page.getByText('Patient 3 — High Risk')).toBeVisible()
  })

  test('each card shows risk level and description', async ({ page }) => {
    await page.goto('/')
    await expect(page.getByText('Post-surgical recovery')).toBeVisible({ timeout: 10000 })
    await expect(page.getByText(/pneumonia/i)).toBeVisible()
    await expect(page.getByText(/Septic shock/i)).toBeVisible()
  })

  test('ClinIQ branding is visible', async ({ page }) => {
    await page.goto('/')
    await expect(page.getByText('ClinIQ')).toBeVisible({ timeout: 5000 })
  })

  test('clicking Patient 1 navigates to monitor dashboard', async ({ page }) => {
    await page.goto('/')
    await page.getByText('Patient 1 — Low Risk').click()
    await expect(page).toHaveURL(/\/monitor\/1/, { timeout: 10000 })
  })

  test('clicking Patient 3 navigates to high-risk dashboard', async ({ page }) => {
    await page.goto('/')
    await page.getByText('Patient 3 — High Risk').click()
    await expect(page).toHaveURL(/\/monitor\/3/, { timeout: 10000 })
  })

  test('header shows live clock', async ({ page }) => {
    await page.goto('/')
    // Time in HH:MM:SS format
    await expect(page.locator('text=/\\d{2}:\\d{2}:\\d{2}/')).toBeVisible({ timeout: 5000 })
  })

  test('dashboard header shows patient name after navigating to Patient 1', async ({ page }) => {
    await page.goto('/')
    await page.getByText('Patient 1 — Low Risk').click()
    await expect(page).toHaveURL(/\/monitor\/1/, { timeout: 10000 })
    // HeaderBar / PatientPill shows patient name after data loads
    await expect(page.getByText('Patient 1 — Low Risk').first()).toBeVisible({ timeout: 10000 })
  })
})
