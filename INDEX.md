# Test index

Auto-generated from `experiments/registry.py` (`python -m experiments index`).

## `hamiltonian_flow.ipynb` — No normalizing flow: parametric DF + physics-bias tests

| Test | Studies | True $f_0$ | Code | Status |
|------|---------|-----------|------|--------|
| 1 | Frozen biased Gaussian DF, only h free -> fit breaks | single Gaussian (frozen, biased) | `experiments/exp_01_frozen_biased_gauss.py` | pending |
| 1b | r^2 bias-asymmetry verification + exact R_eff | single Gaussian | `experiments/exp_01b_bias_asymmetry.py` | pending |
| 2 | One dataset, two time-dep corrections swept over alpha | single Gaussian + satellite | `experiments/exp_02_timedep_corrections.py` | pending |
| 2a | Box-envelope satellite correction | single Gaussian + satellite | `experiments/exp_02a_box_satellite.py` | pending |
| 2b | Fourier h(t) correction | single Gaussian + h(t) | `experiments/exp_02b_fourier_ht.py` | pending |
| 3 | Free Gaussian DF + 10%-capped NN potential | single Gaussian (free mu,sigma) + 10% NN | `experiments/exp_03_free_gauss_nn.py` | pending |
| 3b | Single-Gaussian DF + time-dependent NN potential | single Gaussian + time-dep NN | `experiments/exp_03b_free_gauss_timedep_nn.py` | pending |

## `NF_hdm_fit__gaussian.ipynb` — NF h_dm fit, true f0 = centred single Gaussian (mean 0)

| Test | Studies | True $f_0$ | Code | Status |
|------|---------|-----------|------|--------|
| PG | Profile likelihood by NF kind & #params (+ analytic linear) | centred Gaussian | `experiments/exp_prof_gaussian.py` | migrated |
| 9 | Flow AS f0 optimised jointly with h (+ why: profile) | centred Gaussian | `experiments/exp_09_joint_flow_h.py` | migrated |
| 10 | Flexible NF f0 recovers h + honest error bars | centred Gaussian | `experiments/exp_10_flexible_nf.py` | pending |
| 10c | Bias sweep: fitted h vs sigma_w pdf-mismatch, 4 f0 models | centred Gaussian (sigma_w mismatch sweep) | `experiments/exp_10c_bias_sweep.py` | pending |
| 5 | When does the NF actually help recover h? | centred Gaussian (framework) | `experiments/exp_05_when_nf_helps.py` | pending |
| 6 | Satellite-biased MLE: does the NN potential-cap matter? | centred Gaussian + satellite | `experiments/exp_06_sat_nn_cap.py` | pending |
| 7 | Harmonic vs anharmonic regime diagnostic | centred Gaussian (framework) | `experiments/exp_07_harmonic_regime.py` | pending |
| A | Linear bijection F(u)=a*u+b (4 params) recovers a,b,h | centred Gaussian | `experiments/exp_A_linear_bijection.py` | migrated |
| B | Tiny 8-parameter NN bijection recovers h | centred Gaussian | `experiments/exp_B_tiny_nn.py` | migrated |
| C | Bijection-size sweep: h constraint vs #params (4..35k) | centred Gaussian | `experiments/exp_C_size_sweep.py` | migrated |
| CV | init-guess<->fitted-h correlation is under-convergence (step sweep) | centred Gaussian | `experiments/exp_conv_gaussian.py` | migrated |

## `NF_hdm_fit__gaussian__time_parameter.ipynb` — NF h_dm fit + free total flow time Delta t (Gaussian f0)

| Test | Studies | True $f_0$ | Code | Status |
|------|---------|-----------|------|--------|
| PGT | Joint (h_dm, Delta t) fit: total flow time as a free parameter | centred Gaussian | `experiments/exp_prof_gaussian_dt.py` | migrated |

## `NF_hdm_fit__shifted_mean.ipynb` — NF h_dm fit, true f0 = shifted / different-mean Gaussians

| Test | Studies | True $f_0$ | Code | Status |
|------|---------|-----------|------|--------|
| PSM | Profile likelihood by NF kind + distribution fit | single Gaussian in w, mean 30 | `experiments/exp_prof_shifted.py` | migrated |
| 12 | Non-Gaussian f0: symmetric bimodal (two means) | 2 Gaussians in w, means +/-35, same width | `experiments/exp_12_bimodal_mean.py` | migrated |
| 13 | Shifted-mean Gaussian f0 (non-equilibrium offset) | single Gaussian in w, mean 30 (bulk streaming) | `experiments/exp_13_shifted_mean.py` | migrated |

## `NF_hdm_fit__two_widths.ipynb` — NF h_dm fit, true f0 = two Gaussians in w (mean 0, widths 15 & 60)

| Test | Studies | True $f_0$ | Code | Status |
|------|---------|-----------|------|--------|
| P2W | Profile likelihood by NF kind + distribution fit | 2 Gaussians in w, mean 0, widths 15 & 60 | `experiments/exp_prof_two_widths.py` | migrated |
| 4 | data -> NF -> Hamiltonian flow framework (10% NN) | double Gaussian, mean 0 (cold sw12 + hot sw45) | `experiments/exp_04_framework_double_gauss.py` | pending |
| 11 | Non-Gaussian f0: sharp core + heavy wings | 2 Gaussians in w, same mean 0, widths 30 & 40 | `experiments/exp_11_core_wings.py` | migrated |

## `NF_hdm_fit__three_widths.ipynb` — NF h_dm fit, true f0 = three Gaussians in w (mean 0, widths 15/40/90)

| Test | Studies | True $f_0$ | Code | Status |
|------|---------|-----------|------|--------|
| P3W | Profile likelihood by NF kind + distribution fit | 3 Gaussians in w, mean 0, widths 15/40/90 | `experiments/exp_prof_three_widths.py` | migrated |
