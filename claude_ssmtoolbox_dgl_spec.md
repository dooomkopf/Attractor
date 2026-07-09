# Claude Implementation Spec: `ssmtoolbox_dgl.py`

You are the implementer. Work in `/home/hz/Data/Attractor` and edit files directly.
Create a new script:

- `/home/hz/Data/Attractor/ssmtoolbox_dgl.py`

Do not touch unrelated files. Treat existing user changes as authoritative and do not revert anything.

## Goal

Build a model-driven analyzer for the LPPL/Wang DGL that follows the workflow and decision logic in:

- `/home/hz/Data/Attractor/SSMToolHaller.md`
- `/home/hz/Data/Attractor/SSMToolHaller_quickref.md`

as closely as possible, but honestly reports where direct SSMtool V1.0 applicability fails.

## Context

- Exact DGL code: `/home/hz/Data/LPPL-forced/LPPL-attractor/lpplattr02_ode.py`
- Parameters: `/home/hz/Data/LPPL-forced/LPPL-attractor/lpplattr02_params.py`
- The user wants the REAL DGL analysis, not another data-driven SSMLearn fit.
- Mandatory ingredients that must remain represented in the analysis:
  - LPPL core
  - Wang coupling
  - diffusion/relaxation term in `z`
- If regularization/surrogate modes are offered, they must be explicit and justified, not silent simplifications.

## What to Implement

### 1. CLI script

Create a CLI script that analyzes the DGL in an SSMtool-style workflow.

### 2. Analysis modes

Support at least these modes:

- `exact`
  - analyze the exact/local model structure
  - report SSMtool obstacles precisely
- `regularized`
  - keep the model structure
  - replace the non-analytic `y2*|y2|^(M-1)` term by an explicit smooth regularization such as
    `y2*(eps^2 + y2^2)^((M-1)/2)`
  - make `eps` configurable

### 3. Replicate the SSMtool workflow conceptually

The script should, as far as meaningful for this DGL, do the following:

1. identify the vector field used for analysis
2. identify equilibrium candidates
   - at least the origin explicitly
3. compute the local Jacobian at the equilibrium
4. compute eigenvalues and eigenvectors
5. check asymptotic stability / hyperbolicity relevant to SSMtool
6. enumerate candidate 2D master subspaces where meaningful
7. compute/report spectral quotient where meaningful
8. run external/internal resonance-style checks where meaningful
9. assess analyticity / smoothness assumptions
10. assess autonomy / forcing issues
11. assess mechanical-form compatibility with SSMtool V1.0
12. output a structured verdict:
    - direct SSMtool path possible or blocked
    - why

### 4. Outputs

The script should produce:

- high-signal terminal output
- optional Markdown report
- optional JSON summary

in the working directory.

### 5. Implementation style

- keep it self-contained and readable
- prefer standard Python + `numpy` + `sympy`
- `scipy` is acceptable for numerical fallback
- do not fake a full `compute_SSM` solver if the model is blocked
- if direct SSMtool use is impossible, report that explicitly and stop at the precise boundary
- if useful, include a clearly marked surrogate/preparation section for what would be needed next

### 6. Practical CLI options

Use good judgment, but include practical options along these lines:

- `--mode exact|regularized`
- `--eps <float>`
- `--report <path>`
- `--json <path>`
- `--order <int>`
- `--verbose`

### 7. Important review criteria

- match the reasoning in `SSMToolHaller.md`, especially the LPPL applicability discussion
- do not silently reinterpret the DGL as if it were already in valid SSMtool mechanical form
- preserve the distinction between exact analysis and surrogate/regularized analysis
- make the script useful even if the answer is “SSMtool direct path is blocked”

## Response after coding

Print a concise summary with:

- changed file paths
- implemented CLI/options
- exact limitations still open
- assumptions you had to make
