"""The single index of every test: id -> where its code lives, which notebook
shows it, the true DF it studies, and a one-line result.

`status`:  'migrated' = has an exp_*.py module here;  'pending' = still only in
the archived notebooks under old/ (roadmap).
"""

REGISTRY = {
    # ---- no normalizing flow: parametric / physics-bias tests --------------
    "1":   dict(module="exp_01_frozen_biased_gauss", notebook="hamiltonian_flow",
                true_df="single Gaussian (frozen, biased)", status="pending",
                title="Frozen biased Gaussian DF, only h free -> fit breaks",
                result=""),
    "1b":  dict(module="exp_01b_bias_asymmetry", notebook="hamiltonian_flow",
                true_df="single Gaussian", status="pending",
                title="r^2 bias-asymmetry verification + exact R_eff", result=""),
    "2":   dict(module="exp_02_timedep_corrections", notebook="hamiltonian_flow",
                true_df="single Gaussian + satellite", status="pending",
                title="One dataset, two time-dep corrections swept over alpha", result=""),
    "2a":  dict(module="exp_02a_box_satellite", notebook="hamiltonian_flow",
                true_df="single Gaussian + satellite", status="pending",
                title="Box-envelope satellite correction", result=""),
    "2b":  dict(module="exp_02b_fourier_ht", notebook="hamiltonian_flow",
                true_df="single Gaussian + h(t)", status="pending",
                title="Fourier h(t) correction", result=""),
    "3":   dict(module="exp_03_free_gauss_nn", notebook="hamiltonian_flow",
                true_df="single Gaussian (free mu,sigma) + 10% NN", status="pending",
                title="Free Gaussian DF + 10%-capped NN potential", result=""),
    "3b":  dict(module="exp_03b_free_gauss_timedep_nn", notebook="hamiltonian_flow",
                true_df="single Gaussian + time-dep NN", status="pending",
                title="Single-Gaussian DF + time-dependent NN potential", result=""),

    # ---- NF, true DF = centred Gaussian (mean 0) ---------------------------
    "PG":  dict(module="exp_prof_gaussian", notebook="NF_hdm_fit__gaussian",
                true_df="centred Gaussian", status="migrated",
                title="Profile likelihood by NF kind & #params (+ analytic linear)",
                result=""),
    "PGT": dict(module="exp_prof_gaussian_dt", notebook="NF_hdm_fit__gaussian__time_parameter",
                true_df="centred Gaussian", status="migrated",
                title="Joint (h_dm, Delta t) fit: total flow time as a free parameter",
                result="dt FIXED at 0.5 Myr, free parameter = real-valued step "
                       "count N (Delta t = N*dt). NLL(Delta t) is an alias comb "
                       "(~16 local minima, 100-300 nat barriers) so no local "
                       "optimiser works; tuned fitter = global de-aliasing sweep "
                       "+ full-batch descent + polish -> same optimum from every "
                       "start. See experiments/hyperopt_time.py."),
    "9":   dict(module="exp_09_joint_flow_h", notebook="NF_hdm_fit__gaussian",
                true_df="centred Gaussian", status="migrated",
                title="Flow AS f0 optimised jointly with h (+ why: profile)",
                result=""),
    "10":  dict(module="exp_10_flexible_nf", notebook="NF_hdm_fit__gaussian",
                true_df="centred Gaussian", status="pending",
                title="Flexible NF f0 recovers h + honest error bars", result=""),
    "10c": dict(module="exp_10c_bias_sweep", notebook="NF_hdm_fit__gaussian",
                true_df="centred Gaussian (sigma_w mismatch sweep)", status="pending",
                title="Bias sweep: fitted h vs sigma_w pdf-mismatch, 4 f0 models", result=""),
    "5":   dict(module="exp_05_when_nf_helps", notebook="NF_hdm_fit__gaussian",
                true_df="centred Gaussian (framework)", status="pending",
                title="When does the NF actually help recover h?", result=""),
    "6":   dict(module="exp_06_sat_nn_cap", notebook="NF_hdm_fit__gaussian",
                true_df="centred Gaussian + satellite", status="pending",
                title="Satellite-biased MLE: does the NN potential-cap matter?", result=""),
    "7":   dict(module="exp_07_harmonic_regime", notebook="NF_hdm_fit__gaussian",
                true_df="centred Gaussian (framework)", status="pending",
                title="Harmonic vs anharmonic regime diagnostic", result=""),
    "A":   dict(module="exp_A_linear_bijection", notebook="NF_hdm_fit__gaussian",
                true_df="centred Gaussian", status="migrated",
                title="Linear bijection F(u)=a*u+b (4 params) recovers a,b,h",
                result=""),
    "B":   dict(module="exp_B_tiny_nn", notebook="NF_hdm_fit__gaussian",
                true_df="centred Gaussian", status="migrated",
                title="Tiny 8-parameter NN bijection recovers h", result=""),
    "C":   dict(module="exp_C_size_sweep", notebook="NF_hdm_fit__gaussian",
                true_df="centred Gaussian", status="migrated",
                title="Bijection-size sweep: h constraint vs #params (4..35k)", result=""),
    "CV":  dict(module="exp_conv_gaussian", notebook="NF_hdm_fit__gaussian",
                true_df="centred Gaussian", status="migrated",
                title="init-guess<->fitted-h correlation is under-convergence (step sweep)",
                result=""),

    # ---- NF, true DF = two Gaussians, mean 0, different width --------------
    "P2W": dict(module="exp_prof_two_widths", notebook="NF_hdm_fit__two_widths",
                true_df="2 Gaussians in w, mean 0, widths 15 & 60", status="migrated",
                title="Profile likelihood by NF kind + distribution fit", result=""),
    "4":   dict(module="exp_04_framework_double_gauss", notebook="NF_hdm_fit__two_widths",
                true_df="double Gaussian, mean 0 (cold sw12 + hot sw45)", status="pending",
                title="data -> NF -> Hamiltonian flow framework (10% NN)", result=""),
    "11":  dict(module="exp_11_core_wings", notebook="NF_hdm_fit__two_widths",
                true_df="2 Gaussians in w, same mean 0, widths 30 & 40", status="migrated",
                title="Non-Gaussian f0: sharp core + heavy wings", result=""),

    # ---- NF, true DF = three Gaussians in w, mean 0, different widths ------
    "P3W": dict(module="exp_prof_three_widths", notebook="NF_hdm_fit__three_widths",
                true_df="3 Gaussians in w, mean 0, widths 15/40/90", status="migrated",
                title="Profile likelihood by NF kind + distribution fit", result=""),

    # ---- NF, true DF = shifted / different mean ----------------------------
    "PSM": dict(module="exp_prof_shifted", notebook="NF_hdm_fit__shifted_mean",
                true_df="single Gaussian in w, mean 30", status="migrated",
                title="Profile likelihood by NF kind + distribution fit", result=""),
    "12":  dict(module="exp_12_bimodal_mean", notebook="NF_hdm_fit__shifted_mean",
                true_df="2 Gaussians in w, means +/-35, same width", status="migrated",
                title="Non-Gaussian f0: symmetric bimodal (two means)", result=""),
    "13":  dict(module="exp_13_shifted_mean", notebook="NF_hdm_fit__shifted_mean",
                true_df="single Gaussian in w, mean 30 (bulk streaming)", status="migrated",
                title="Shifted-mean Gaussian f0 (non-equilibrium offset)", result=""),
}

NOTEBOOKS = {
    "hamiltonian_flow":        "No normalizing flow: parametric DF + physics-bias tests",
    "NF_hdm_fit__gaussian":    "NF h_dm fit, true f0 = centred single Gaussian (mean 0)",
    "NF_hdm_fit__gaussian__time_parameter":
                               "NF h_dm fit + free total flow time Delta t (Gaussian f0)",
    "NF_hdm_fit__shifted_mean":"NF h_dm fit, true f0 = shifted / different-mean Gaussians",
    "NF_hdm_fit__two_widths":  "NF h_dm fit, true f0 = two Gaussians in w (mean 0, widths 15 & 60)",
    "NF_hdm_fit__three_widths":"NF h_dm fit, true f0 = three Gaussians in w (mean 0, widths 15/40/90)",
}
