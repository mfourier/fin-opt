# FinOpt Development Roadmap

> Detailed development plan for advancing the FinOpt platform

Last updated: February 7, 2026

---

## Current State Summary

### Implemented Components

| Component | Status | Description |
|-----------|--------|-------------|
| **Mathematical Core** | ✅ Complete | CVaR optimization, Monte Carlo simulation, affine wealth dynamics |
| **Income Modeling** | ✅ Complete | Fixed + variable income with seasonality, raises, floor/cap |
| **Withdrawal Support** | ✅ Complete | Scheduled deterministic + stochastic withdrawals |
| **Goals Framework** | ✅ Complete | Intermediate + terminal goals with confidence levels |
| **FastAPI Backend** | ✅ Complete | /simulate, /optimize endpoints with background jobs |
| **React Frontend** | ✅ Complete | Profiles, Scenarios, Results pages with full CRUD |
| **Supabase Integration** | ✅ Complete | Auth, database, real-time subscriptions |
| **Docker Deployment** | ✅ Complete | Containerized services ready for production |
| **Results Visualization** | ✅ Complete | Allocation charts, goal progress, interactive heatmaps |
| **Form Validation** | ✅ Complete | Client-side validation with error feedback |
| **Toast Notifications** | ✅ Complete | Success/error feedback for all operations |

### Known Limitations

1. **No historical validation**: Optimization uses theoretical returns, not backtested
2. **Single currency**: All values assumed in same currency (CLP by default)
3. **No tax modeling**: Returns are pre-tax
4. **No transaction costs**: Rebalancing is frictionless
5. **No wealth trajectory simulation**: Results show allocation but not projected wealth paths

---

## Recently Completed (February 2026)

### ✅ Results Visualization Improvements

**Delivered:**
- `AllocationChart.tsx` - Stacked area chart with Recharts showing allocation over time
- `GoalProgressCard.tsx` - Visual progress bars for each goal with satisfaction status
- Enhanced `AllocationHeatmap.tsx`:
  - Account-specific color coding (blue, green, amber, red, purple, pink)
  - Toggle between heatmap and stacked bar views
  - Interactive tooltips on hover
  - Summary statistics per account (average, range, trend)
- `ResultsPage.tsx` improvements:
  - Tab navigation (Overview, Allocation, Goals)
  - Better summary cards with formatted horizon (e.g., "2y 5m")
  - CSV export for allocation policy
  - Improved loading and error states

### ✅ Form Validation System

**Delivered:**
- `validation.ts` - Validation functions for profiles and scenarios
- `FormField.tsx` - Reusable form components with error styling
- `ValidationSummary` component for displaying all errors
- Field-level error highlighting in forms
- Validation for:
  - Required fields
  - Numeric ranges (returns, volatility, confidence)
  - Date validation for intermediate goals
  - Duplicate account detection

### ✅ Error Handling & User Feedback

**Delivered:**
- `Toast.tsx` - Complete toast notification system
- `ToastProvider` context wrapper
- `useToast()` hook for triggering notifications
- Toast types: success, error, warning, info
- Auto-dismiss with configurable duration
- Integrated in ProfilesPage and ScenariosPage for all CRUD operations

---

## Phase 1: Core Enhancements (Priority: High)

### 1.1 Wealth Trajectory Visualization

**Goal**: Show projected wealth paths from simulation results.

**Tasks**:
- [ ] Modify backend to save wealth percentiles (P10, P25, P50, P75, P90) in results
- [ ] Create `WealthTrajectoryChart.tsx` with fan chart visualization
- [ ] Show goal thresholds as horizontal lines on the chart
- [ ] Add account selector to view individual account trajectories

**Backend changes needed** (`api/services/optimization.py`):
```python
# After optimization, re-simulate with optimal X to get wealth trajectories
sim_result = model.simulate_from_optimization(opt_result, n_sims=n_sims, seed=seed)

# Compute percentiles
wealth_percentiles = {
    'p10': np.percentile(sim_result.wealth, 10, axis=0).tolist(),
    'p25': np.percentile(sim_result.wealth, 25, axis=0).tolist(),
    'p50': np.percentile(sim_result.wealth, 50, axis=0).tolist(),
    'p75': np.percentile(sim_result.wealth, 75, axis=0).tolist(),
    'p90': np.percentile(sim_result.wealth, 90, axis=0).tolist(),
}
```

**Effort**: 1-2 weeks

---

### 1.2 Historical Backtesting Module

**Goal**: Validate allocation strategies against historical market data.

**Tasks**:
- [ ] Create `src/finopt/backtest.py` module
- [ ] Integrate historical return data (CSV upload or yfinance)
- [ ] Implement walk-forward validation
- [ ] Compare optimized vs naive strategies (60/40, equal weight)
- [ ] Add backtesting page in frontend
- [ ] Add `/backtest` endpoint to API

**Technical approach**:
```python
class Backtester:
    def __init__(self, historical_returns: pd.DataFrame):
        """
        Args:
            historical_returns: DataFrame with columns = account names,
                               index = dates, values = monthly returns
        """

    def run(self, allocation_policy: np.ndarray, start_date: date) -> BacktestResult:
        """Execute the allocation policy on historical data."""

    def compare_strategies(self, strategies: dict[str, np.ndarray]) -> pd.DataFrame:
        """Compare multiple allocation strategies."""
```

**Effort**: 3-4 weeks

---

### 1.3 Robust Optimization

**Goal**: Handle uncertainty in expected returns and volatilities.

**Tasks**:
- [ ] Implement ellipsoidal uncertainty sets for (μ, σ)
- [ ] Add robust counterpart formulation to CVaROptimizer
- [ ] Create uncertainty level selector in scenario form
- [ ] Document mathematical formulation

**Technical approach**:
```python
class RobustCVaROptimizer(CVaROptimizer):
    def __init__(self, uncertainty_radius: float = 0.1, ...):
        """
        Args:
            uncertainty_radius: Size of uncertainty set (0 = nominal, 1 = very conservative)
        """

    def _add_robust_constraints(self, ...):
        """Add worst-case constraints for return uncertainty."""
```

**Effort**: 2-3 weeks

---

## Phase 2: Platform Features (Priority: Medium)

### 2.1 Goal Templates

**Goal**: Pre-configured goals for common use cases.

**Templates to implement**:
1. **Emergency Fund**: 6 months expenses in low-volatility account
2. **House Down Payment**: Fixed amount by specific date
3. **Retirement**: Terminal wealth goal with inflation adjustment
4. **Education Fund**: Intermediate goal for tuition payments
5. **Vacation Fund**: Annual withdrawal pattern

**Implementation**:
```typescript
// web/src/lib/goalTemplates.ts
export const goalTemplates = {
  emergencyFund: {
    name: "Emergency Fund",
    description: "6 months of expenses in safe account",
    createGoal: (monthlyExpenses: number, accountName: string) => ({
      account: accountName,
      threshold: monthlyExpenses * 6,
      confidence: 0.95,
    })
  },
  // ...
};
```

**Effort**: 1 week

---

### 2.2 Scenario Comparison

**Goal**: Allow users to compare multiple scenarios side-by-side.

**Tasks**:
- [ ] Create comparison page in frontend
- [ ] Store multiple results per scenario (history)
- [ ] Add comparison metrics (horizon, terminal wealth, goal margins)
- [ ] Visual diff of allocation policies

**Database changes**:
```sql
ALTER TABLE results ADD COLUMN version INTEGER DEFAULT 1;
CREATE INDEX idx_results_scenario_version ON results(job_id, version);
```

**Effort**: 2 weeks

---

### 2.3 What-If Analysis

**Goal**: Sensitivity analysis for key parameters.

**Features**:
- Slider to vary confidence levels
- Impact of different contribution rates
- Effect of changing return assumptions
- Tornado diagram for parameter sensitivity

**Tasks**:
- [ ] Add sensitivity endpoint to API
- [ ] Create interactive sensitivity UI
- [ ] Cache results for smooth slider experience
- [ ] Generate summary insights

**Effort**: 2-3 weeks

---

### 2.4 PDF Report Generation

**Goal**: Professional exportable reports.

**Report sections**:
1. Executive Summary (horizon, key metrics)
2. Profile Configuration (income, accounts)
3. Scenario Details (goals, parameters)
4. Optimization Results (allocation policy chart)
5. Risk Analysis (goal probabilities)
6. Appendix (methodology, assumptions)

**Technical approach**:
- Use `reportlab` or `weasyprint` in backend
- New endpoint: `POST /reports/{job_id}`
- Store generated PDFs in Supabase Storage

**Effort**: 2 weeks

---

## Phase 3: Advanced Capabilities (Priority: Low)

### 3.1 Tax-Aware Optimization

**Goal**: Optimize after-tax wealth accumulation.

**Considerations**:
- Different tax rates per account type (taxable, tax-deferred, tax-free)
- Capital gains vs ordinary income
- Tax-loss harvesting opportunities
- Jurisdiction-specific rules

**Effort**: 4-6 weeks

---

### 3.2 Multi-Currency Support

**Goal**: Handle portfolios across different currencies.

**Tasks**:
- [ ] Add currency field to accounts
- [ ] Integrate exchange rate data
- [ ] Model currency risk in returns
- [ ] Display in user's base currency

**Effort**: 3-4 weeks

---

### 3.3 Real-Time Portfolio Monitoring

**Goal**: Track actual portfolio performance against plan.

**Features**:
- Manual or API-based position entry
- Variance tracking (planned vs actual allocation)
- Rebalancing recommendations
- Progress toward goals

**Effort**: 4-6 weeks

---

### 3.4 Public API for External Tools

**Goal**: Enable integration with other financial tools.

**Endpoints**:
```
POST /api/v1/optimize     # Synchronous optimization (small problems)
GET  /api/v1/scenarios    # List user scenarios
GET  /api/v1/results/{id} # Fetch optimization results
POST /api/v1/simulate     # Run simulation without optimization
```

**Requirements**:
- API key authentication
- Rate limiting
- OpenAPI documentation

**Effort**: 2 weeks

---

## Phase 4: Enterprise Features (Priority: Future)

### 4.1 Multi-Tenant Architecture
- Workspace/organization support
- Role-based access control
- Shared scenarios with permissions

### 4.2 Advisor Dashboard
- Client management
- Bulk scenario management
- White-label customization

### 4.3 Compliance & Audit
- Action logging
- Regulatory reporting templates
- Data retention policies

---

## Technical Debt & Maintenance

### Testing Improvements
- [ ] Increase test coverage to >80%
- [ ] Add E2E tests with Playwright
- [ ] Property-based testing for optimization
- [ ] Performance benchmarks

### Documentation
- [ ] API documentation with OpenAPI/Swagger
- [ ] User guide with screenshots
- [ ] Video tutorials
- [ ] Contribution guidelines

### Infrastructure
- [ ] Implement request rate limiting
- [ ] Add structured logging throughout
- [ ] Set up monitoring and alerting
- [ ] Performance optimization for large horizons

---

## Prioritized Next Steps

### Immediate (Next 2 weeks)
1. **Wealth Trajectory Chart**: Add fan chart showing projected wealth percentiles
2. **Goal Templates**: Implement top 3 most common templates
3. **Profile Validation**: Add validation to ProfilesPage (matching ScenariosPage)

### Short-term (Next 1-2 months)
1. **Historical Backtesting**: MVP with CSV upload
2. **PDF Reports**: Basic report generation
3. **Scenario Comparison**: Side-by-side analysis

### Medium-term (Next 3-6 months)
1. **Robust Optimization**: Uncertainty sets for returns
2. **What-If Analysis**: Sensitivity sliders
3. **Public API**: REST API with authentication

---

## Success Metrics

| Metric | Current | Target (6 months) |
|--------|---------|-------------------|
| Test Coverage | ~60% | >80% |
| API Response Time (p95) | ~2s | <500ms for small scenarios |
| Frontend Lighthouse Score | - | >90 |
| Active Users | - | Track after launch |
| Scenarios Created | - | Track after launch |

---

## Changelog

### 2026-02-07
- ✅ Completed Results Visualization (AllocationChart, GoalProgressCard, enhanced heatmap)
- ✅ Completed Form Validation system
- ✅ Completed Toast Notification system
- Updated priorities: Wealth Trajectory is now the top priority
- Added Goal Templates to immediate priorities

### 2026-02-01
- Initial roadmap created
- Platform deployed to Render

---

## Resources & References

### Mathematical References
- Rockafellar & Uryasev (2000) - CVaR Optimization
- Ben-Tal et al. (2009) - Robust Optimization
- Boyd & Vandenberghe (2004) - Convex Optimization

### Technical References
- CVXPY Documentation
- FastAPI Best Practices
- React Query Patterns
- Recharts Documentation

---

*This roadmap is a living document. Priorities may shift based on user feedback and resource availability.*
