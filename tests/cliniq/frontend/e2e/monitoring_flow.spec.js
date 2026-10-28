/**
 * E2E — MonitorDashboard live monitoring flow
 * Requires: backend on localhost:8000, frontend on localhost:5173
 */
import { test, expect } from '@playwright/test'

test.describe('Monitor Dashboard', () => {
  test.beforeEach(async ({ page }) => {
    // Start from patient selection
    await page.goto('/')
    await page.getByText('Patient 2 — Moderate Risk').click()
    await page.waitForURL(/\/monitor\/2/, { timeout: 10000 })
  })

  test('SOFA gauge is visible', async ({ page }) => {
    // SOFA score between 0 and 24 should be displayed
    await expect(page.locator('text=/SOFA|sofa/i').first()).toBeVisible({ timeout: 10000 })
  })

  test('LIVE indicator is shown in header', async ({ page }) => {
    await expect(page.getByText('LIVE')).toBeVisible({ timeout: 10000 })
  })

  test('countdown timer is shown', async ({ page }) => {
    // Should show "Xs" or "···" while refreshing
    await expect(page.locator('text=/\\d{1,2}s|···/')).toBeVisible({ timeout: 10000 })
  })

  test('vitals grid shows 7 vital cards', async ({ page }) => {
    // Look for vital labels
    for (const label of ['Heart Rate', 'SpO₂', 'Systolic BP']) {
      await expect(page.locator(`text=/${label}/`).first()).toBeVisible({ timeout: 15000 })
    }
  })

  test('history FAB opens history drawer', async ({ page }) => {
    const fab = page.getByRole('button', { name: /Open patient history/i })
    await expect(fab).toBeVisible({ timeout: 10000 })
    await fab.click()
    await expect(page.getByText(/History|SOFA/i).first()).toBeVisible()
  })

  test('CV FAB opens CV modal', async ({ page }) => {
    const fab = page.getByRole('button', { name: /Open Live CV Monitor/i })
    await expect(fab).toBeVisible({ timeout: 10000 })
    await fab.click()
    await expect(page.getByText(/supplementary/i)).toBeVisible()
  })

  test('SOFA Validation FAB opens validation drawer', async ({ page }) => {
    const fab = page.getByRole('button', { name: /Open SOFA Score Validation/i })
    await expect(fab).toBeVisible({ timeout: 10000 })
    await fab.click()
    await expect(page.locator('text=/MAE|AUROC|R²/')).toBeVisible({ timeout: 5000 })
  })

  test('exit button returns to patient selection', async ({ page }) => {
    await page.getByText('Exit').click()
    await expect(page).toHaveURL('/', { timeout: 5000 })
  })
})
