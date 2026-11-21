import { render, screen, fireEvent } from '@testing-library/react'
import SOFAValidationDrawer from '@/components/validation/SOFAValidationDrawer.jsx'

// Shape mirrors models/training_metadata.json — the drawer must show these
// values (not hard-coded numbers) so it always describes the loaded model.
const META = {
  final_mae: 1.8247,
  final_r2: 0.4725,
  train_samples: 41798,
  test_samples: 10450,
  evaluation_limitation: 'Retrospective MIMIC-III results.',
  risk_classification_metrics: {
    risk_bands: {
      labels: ['Low (<5)', 'Moderate (5-<10)', 'High (>=10)'],
      confusion_matrix: [[6617, 305, 7], [2159, 669, 91], [112, 203, 287]],
      per_class: { 'High (>=10)': { recall: 0.4767 } },
    },
    high_risk_alert: {
      predicted_sofa_threshold: 7.0,
      auroc: 0.9254, pr_auc: 0.6456,
      recall: 0.6711, specificity: 0.9634,
      false_negative: 198, false_positive: 360,
    },
  },
}

const predictData = { patient_id: 'P1', sofa_score: 6.2, _training_meta: META }

const openDrawer = (props = {}) =>
  render(<SOFAValidationDrawer open={true} onClose={() => {}} predictData={predictData} {...props} />)

describe('SOFAValidationDrawer', () => {
  it('does not render content when closed', () => {
    const { container } = render(<SOFAValidationDrawer open={false} onClose={() => {}} />)
    expect(container.textContent).not.toMatch(/SOFA Prediction/i)
  })

  it('renders when open=true', () => {
    openDrawer()
    expect(document.body.textContent).toMatch(/SOFA|Model/i)
  })

  it('shows MAE from training metadata', () => {
    openDrawer()
    expect(screen.getByText('1.8247')).toBeInTheDocument()
  })

  it('shows R² from training metadata', () => {
    openDrawer()
    expect(screen.getByText('0.4725')).toBeInTheDocument()
  })

  it('shows AUROC from training metadata', () => {
    openDrawer()
    expect(screen.getByText('0.9254')).toBeInTheDocument()
  })

  it('shows PR-AUC from training metadata', () => {
    openDrawer()
    expect(screen.getByText('0.6456')).toBeInTheDocument()
  })

  it('computes 3-class accuracy from the confusion matrix', () => {
    openDrawer()
    // (6617 + 669 + 287) / 10450 = 72.5%
    expect(document.body.textContent).toMatch(/72\.5%/)
  })

  it('shows the test and total sample counts', () => {
    openDrawer()
    expect(document.body.textContent).toMatch(/10,450 ICU stays of 52,248 total/)
  })

  it('shows the live alert threshold and alert recall', () => {
    openDrawer()
    expect(document.body.textContent).toMatch(/predicted SOFA ≥ 7/)
    expect(screen.getAllByText('67.1%').length).toBeGreaterThan(0)
    expect(screen.getByText('96.3%')).toBeInTheDocument()
  })

  it('shows 3-class High-risk recall from per_class', () => {
    openDrawer()
    expect(screen.getByText('47.7%')).toBeInTheDocument()
  })

  it('shows confusion matrix rows from metadata', () => {
    openDrawer()
    expect(screen.getByText(/True Low Risk/i)).toBeInTheDocument()
    expect(screen.getByText(/True High Risk/i)).toBeInTheDocument()
    expect(screen.getByText('6,617')).toBeInTheDocument()
  })

  it('shows placeholders instead of crashing when metadata is missing', () => {
    render(<SOFAValidationDrawer open={true} onClose={() => {}} predictData={{ sofa_score: 3 }} />)
    expect(screen.getAllByText('—').length).toBeGreaterThan(0)
    expect(document.body.textContent).toMatch(/Confusion matrix not available/)
  })

  it('calls onClose when close button clicked', () => {
    const onClose = vi.fn()
    render(<SOFAValidationDrawer open={true} onClose={onClose} predictData={predictData} />)
    fireEvent.click(screen.getByLabelText(/close panel/i))
    expect(onClose).toHaveBeenCalled()
  })
})
