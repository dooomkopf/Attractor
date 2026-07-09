## Section 1: Files inspected

**SSM/Haller Documentation:**
- `/home/hz/Data/Attractor/SSMToolHaller_new.md` (650 lines)
- `/home/hz/Data/Attractor/SSMLearnHaller_new.md` (158 lines)
- `/home/hz/Data/Attractor/SSMToolHaller_quickref.md` (147 lines)
- `/home/hz/Data/Attractor/SSMLearnHaller_quickref.md` (127 lines)

**BTC Harmonic Analysis:**
- `/home/hz/Data/Attractor/harmonic_test_phase.py` (594 lines)
- `/home/hz/Data/Attractor/harmonic_test_slave.py` (394 lines)
- `/home/hz/Data/Attractor/analyze_residuals/precheck.py` (495 lines)
- `/home/hz/Data/Attractor/analyze_residuals/amplitude.py` (128 lines)
- `/home/hz/Data/Attractor/analyze_residuals/WORKFLOW_OLD_VS_NEW.md` (122 lines)

**Wang System:**
- `/home/hz/Data/Attractor/attractor_2scroll_eq.py` (partial, first 200 lines)
- `/home/hz/Data/Attractor/2-scroll.md` (426 lines)
- `/home/hz/Data/Attractor/analyze_wang/constants.py` (25 lines)
- `/home/hz/Data/Attractor/analyze_wang/system.py` (51 lines)
- `/home/hz/Data/Attractor/analyze_wang/precheck.py` (179 lines)
- `/home/hz/Data/Attractor/analyze_wang/simulate.py` (58 lines)
- `/home/hz/Data/Attractor/analyze_wang/harmonics.py` (85 lines)
- `/home/hz/Data/Attractor/analyze_wang/cli_precheck.py` (29 lines)
- `/home/hz/Data/Attractor/analyze_wang/cli_harmonics.py` (128 lines)

**SSM Repos (directory structure only):**
- `/home/hz/Data/Attractor/SSMTool_jain/` (273MB, 3363 files, OOP MATLAB structure)
- `/home/hz/Data/Attractor/SSMLearnPy/` (Python data-driven SSM)

## Section 2: Findings (ordered by severity)

### CRITICAL: Fundamental SSM applicability mismatch

**Finding:** You're trying to apply SSMtool methodology to systems that violate its core assumptions.

1. **BTC residuals**: Non-autonomous (halvings, time-varying damping), non-polynomial (power law with M=1.071), no stable equilibrium at origin
2. **Wang system**: First-order 3D, no mechanical M,C,K structure, at most 1 complex pair locally
3. **Actual requirement**: Second-order mechanical systems with polynomial nonlinearity, stable equilibrium at origin

**Impact:** Direct SSMtool application impossible. SSMLearn partially applicable but accuracy compromised by smoothness violations.

### CRITICAL: Harmonic mechanism misinterpretation

**Finding:** The new Wang precheck correctly identifies that a 3D system cannot have two independent oscillatory pairs locally.

```python
# analyze_wang/precheck.py:112-116
reason = (
    '3D local linearization can contain at most one complex pair; a separate second oscillatory mode '
    'cannot appear locally as another pair. Quadratic terms (-v*w, +u*w, +u*v) can still generate DC and 2ω content.'
)
```

**But:** Your BTC analysis assumes two independent oscillatory pairs in ssm_dim=4:
```python
# analyze_residuals/precheck.py:203-207
idx_main, idx_sub = identify_modes(eigvals_lin)
```

**Impact:** The "harmonic candidate" interpretation is structurally flawed. In a genuinely 4D reduced system, you can have 2 pairs. But if the true underlying dynamics is 3D quadratic (Wang-like), the second "mode" is a quadratically generated harmonic, not an independent oscillation.

### HIGH: Spectral gap calculation incorrect for growing modes

**Finding:** The SSMtool-style spectral gap assumes all modes decay:

```python
# analyze_residuals/precheck.py:113-115
sigma_in = int(np.fix(np.min(np.real(lambda_all)) / np.max(np.real(lambda_master))))
sigma_out = int(np.fix(np.min(np.real(lambda_slave)) / np.max(np.real(lambda_master))))
```

**But:** Your fitted modes show positive real parts (growth):
- Main mode: Re = +2.706e-01
- Sub mode: Re = +1.349e-01

**Impact:** Spectral gap unavailable. The system is not contracting to an SSM but expanding. This indicates either:
1. The embedding window captures a transient growth phase
2. The system has genuine instability (bubble growth)
3. The linear SSMLearn fit is inadequate

### HIGH: Phase coupling test methodology unsound

**Finding:** `harmonic_test_phase.py` tests phase locking via:
```python
psi_unwrapped = 2.0 * phi_main_u - phi_sub_u
```

**Problem:** This assumes both modes exist as independent linear oscillators. But if the sub-mode is a quadratic slave (as Wang analysis suggests), this test is meaningless - you're comparing a fundamental with its own harmonic.

### MEDIUM: Amplitude support metrics poorly motivated

**Finding:** The amplitude support threshold is arbitrary:
```python
# amplitude.py:27
def _support_threshold(median_main, rel_threshold=0.25)
```

**Issue:** No theoretical basis for 25% threshold. Should be based on noise floor, numerical precision, or physical significance level.

### MEDIUM: Wang precheck oversimplified

**Finding:** The Wang precheck only evaluates local linearization but claims conclusions about global harmonic structure:

```python
# analyze_wang/precheck.py:109
local_second_harmonic_plausible = bool(oscillatory_eqs)
```

**Problem:** Local analysis cannot determine if global nonlinear dynamics generates independent modes via symmetry breaking, heteroclinic connections, or other mechanisms.

## Section 3: What is already solid

1. **Modular architecture**: The new `analyze_residuals/` structure cleanly separates concerns (data, precheck, amplitude, cycles)

2. **Clear parameter distinction**: Explicitly separating "FIXED/VORGEGEBEN" from "LEARNED/AUS DATEN" is excellent methodology

3. **Resonance check implementation**: The scan for near-resonances up to order 4 is properly done:
```python
# precheck.py:63-75
for combo in _mode_combinations(2, max_order):
    combo_ev = combo[0] * lambda_master[0] + combo[1] * lambda_master[1]
```

4. **Wang quadratic channel identification**: Correctly identifies u*v → 2ω mechanism

5. **Embedding budget awareness**: Tracking how many samples remain per halving segment

## Section 4: What is missing / risky

### Missing: Proper SSM existence verification

Neither codebase actually verifies SSM existence conditions:
- Strict spectral gap (all modes stable, masters slowest)
- Non-resonance conditions properly checked
- Smoothness requirements

### Missing: Proper basins/manifolds separation

You're conflating several concepts:
- **SSM**: Tangent to linear eigenspace, requires spectral gap
- **Center manifold**: For critical eigenvalues (Re=0)
- **Inertial manifold**: Global attractor, weaker conditions
- **Slow manifold**: Singular perturbation structure

Your positive growth rates suggest you need center manifold or inertial manifold theory, not SSM.

### Missing: Lyapunov spectrum analysis

You mention λ₁≈0.71/year but don't compute the full spectrum. For a 4D reduced system, you need 4 Lyapunov exponents to characterize dynamics.

### Risky: Forcing SSMLearn on non-smooth dynamics

Power law with M=1.071 violates C^r smoothness at origin. SSMLearn's polynomial fits will have systematic errors near z=0.

### Risky: Over-interpreting Wang analogy

The Wang system is a specific 3D quadratic system. BTC dynamics might be fundamentally higher-dimensional with different nonlinear structure.

## Section 5: Recommended next 3 steps

### Step 1: Determine true dynamical dimension

**Action:** Compute false nearest neighbors (FNN) and Cao's method on the BTC residuals to determine if the dynamics is truly 3D or 4D.

**Implementation:**
```python
from sklearn.neighbors import NearestNeighbors
# Implement FNN test on ctx['pc']
# If dimension < 4, the second "mode" is definitely a harmonic
```

**Expected outcome:** If FNN suggests dim=3, abandon independent second mode hypothesis. If dim≥4, the modes might be independent.

### Step 2: Replace SSM with appropriate framework

**For growing modes (Re>0), implement:**
1. **Center manifold reduction** if some Re≈0
2. **Inertial manifold** via Foias-Sell-Temam theory
3. **Koopman mode decomposition** for empirical spectral analysis

**Concrete code:**
```python
# New module: analyze_residuals/koopman.py
def compute_koopman_modes(data, rank=10):
    """DMD-based Koopman spectral analysis."""
    # This handles growing/decaying modes properly
```

### Step 3: Systematic Wang parameter scan for harmonic mechanism

**Current gap:** You only test default Wang parameters. Need systematic scan to find parameter regime matching BTC behavior.

**Implementation:**
```python
# analyze_wang/scan_harmonic.py
def scan_b_parameter(b_range, measure='harmonic_ratio'):
    """Scan b∈[6.0, 9.0] to find strongest 2:1 resonance."""
    results = []
    for b in b_range:
        sim = simulate_trajectory(b=b)
        harm = analyze_channels(sim)
        results.append((b, harm['pc1']['harmonic_ratio']))
    return results
```

**Why this matters:** The b parameter controls dissipation strength. There might be a sweet spot where 2ω reaches ~50% of fundamental (matching BTC).

### Alternative Step 3: Implement surrogate data test

Before claiming the sub-mode is real, test against IAAFT surrogates:
```python
# analyze_residuals/surrogate.py
def test_harmonic_significance(data, n_surrogates=100):
    """Test if harmonic is stronger than in phase-randomized surrogates."""
    # Generate surrogates preserving spectrum but destroying phase coupling
    # Compare harmonic strength
```
