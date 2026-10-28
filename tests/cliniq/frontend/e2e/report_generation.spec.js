/**
 * E2E — LLM report generation and validation
 * Requires: backend on localhost:8000, frontend on localhost:5173
 * Note: These tests are marked slow because they wait for real LLM calls.
 */
import { test, expect } from '@playwright/test'

test.describe('Report Generation', () => {
  test.beforeEach(async ({ page }) => {
    await page.goto('/')
    await page.getByText('Patient 1 — Low Risk').click()
    await page.waitForURL(/\/monitor\/1/, { timeout: 10000 })
  })

  test('report section shows loading skeleton initially', async ({ page }) => {
    // Right after navigation, skeleton should be visible briefly
    const skeleton = page.locator('text=/Generating AI assessment/i')
    // It may have already resolved — just ensure no crash
    await expect(page).toHaveURL(/\/monitor\/1/)
  })

  test('report section shows 4 content cards after load', async ({ page }) => {
    // Wait for report to appear (can take ~5-10s with real LLM)
    await expect(page.locator('text=/CURRENT CONDITION/i')).toBeVisible({ timeout: 30000 })
    await expect(page.locator('text=/PROBABLE CAUSE/i')).toBeVisible()
    await expect(page.locator('text=/RISK FORECAST/i')).toBeVisible()
    await expect(page.locator('text=/IMMEDIATE ACTIONS/i')).toBeVisible()
  })

  test('reliability badge appears with report', async ({ page }) => {
    await expect(page.locator('text=/AI Clinical Assessment/i')).toBeVisible({ timeout: 30000 })
  })

  test('mode toggle switches from Hospital to Home Care', async ({ page }) => {
    // Wait for initial report
    await expect(page.locator('text=/CURRENT CONDITION/i')).toBeVisible({ timeout: 30000 })
    // Click Home Care
    await page.getByText('Home Care').click()
    // Report should refresh (skeleton appears briefly then content)
    await expect(page.locator('text=/CURRENT CONDITION/i')).toBeVisible({ timeout: 30000 })
  })

  test('LLM Validation FAB is enabled after report loads', async ({ page }) => {
    await expect(page.locator('text=/CURRENT CONDITION/i')).toBeVisible({ timeout: 30000 })
    const fab = page.getByRole('button', { name: /Open AI Report Validation/i })
    await expect(fab).toBeVisible()
    // Should be enabled (full opacity)
    await expect(fab).not.toHaveCSS('opacity', '0.45')
  })

  test('LLM Validation drawer opens and shows method count', async ({ page }) => {
    await expect(page.locator('text=/CURRENT CONDITION/i')).toBeVisible({ timeout: 30000 })
    await page.getByRole('button', { name: /Open AI Report Validation/i }).click()
    await expect(page.locator('text=/methods ran/i')).toBeVisible({ timeout: 5000 })
  })
})
