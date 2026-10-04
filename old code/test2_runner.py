import os
import time as _time
import numpy as np
import matplotlib.pyplot as plt
from fitter import fit_h_pc_dm_nn_mle


def run_test2(sim, h_true, z0_sigma, w0_sigma, t_obs,
              z_obs_test, w_obs_test, h_inits_test, w0_sigma_fit_list,
              w0_sf_arr, h_only_arr,
              h_nn_median, h_nn_lo, h_nn_hi,
              A_nn_median, A_nn_lo, A_nn_hi,
              save_dir="plots"):
    """Run Test 2 (mismatched PDF + time-dependent NN — should RECOVER h_dm)."""

    os.makedirs(save_dir, exist_ok=True)

    test2_nn_results = {}
    n_nn_runs = len(h_inits_test)

    t_start = _time.time()
    for w0_sf in w0_sigma_fit_list:
        ratio = w0_sf / w0_sigma

        print(f"\n{'='*70}")
        print(f"  TEST 2: w0_sigma_fit = {w0_sf:.1f}  ({ratio:.2f}x true)  [time-dependent NN]")
        print(f"    Fitting sigmas: z0_sigma={z0_sigma},  w0_sigma={w0_sf:.1f}  (FIXED z, only w mismatched)")
        print(f"{'='*70}")

        nn_runs = []
        for i, h0 in enumerate(h_inits_test):
            print(f"  NN+h [time-dep] run {i+1}/{n_nn_runs} (h_init={h0})...", flush=True)
            r_nn = fit_h_pc_dm_nn_mle(
                sim, z_obs_test, w_obs_test, t_obs,
                z0_sigma=z0_sigma, w0_sigma=w0_sf,
                h_init=float(h0), use_nn_correction=True, use_satellite=True,
                n_epochs=200, n_sub=500, n_steps=800,
                lr_nn=1e-3, lr_h=5.0, lr_A=1e-2, seed=42,
                time_dependent_nn=True, verbose=False,
            )
            nn_runs.append(r_nn)
            print(f"    -> h={r_nn['h_pc_dm_fitted']:.2f}, A_nn={r_nn['A_nn']:.2e}")
        test2_nn_results[w0_sf] = nn_runs

    print(f"\nTest 2 complete in {(_time.time()-t_start)/60:.1f} min")

    # --- arrays ---
    h_nn_td_all = np.array([[r["h_pc_dm_fitted"] for r in test2_nn_results[s]] for s in w0_sf_arr])
    A_nn_td_all = np.array([[r["A_nn"]            for r in test2_nn_results[s]] for s in w0_sf_arr])

    h_nn_td_median = np.median(h_nn_td_all, axis=1)
    h_nn_td_lo = np.percentile(h_nn_td_all, 16, axis=1)
    h_nn_td_hi = np.percentile(h_nn_td_all, 84, axis=1)

    A_nn_td_median = np.median(A_nn_td_all, axis=1)
    A_nn_td_lo = np.percentile(A_nn_td_all, 16, axis=1)
    A_nn_td_hi = np.percentile(A_nn_td_all, 84, axis=1)

    # --- Plot 1: Test 2 alone ---
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5), dpi=140)
    ax1.axhline(h_true, color='k', ls='--', lw=1.5, label=f'True $h$ = {h_true}')
    ax1.plot(w0_sf_arr, h_only_arr, 'C0o-', lw=2, ms=7, label='h-only (biased)')
    ax1.plot(w0_sf_arr, h_nn_td_median, 'C2D-', lw=2, ms=7, label='Time-dep NN+h (median)')
    ax1.fill_between(w0_sf_arr, h_nn_td_lo, h_nn_td_hi, color='C2', alpha=0.2, label='Time-dep NN+h (16-84%)')
    ax1.set_xlabel('Fitting $\\sigma_w$  (true = 20)')
    ax1.set_ylabel('$h_{\\rm pc,dm}$ fitted [pc]')
    ax1.set_title('Test 2: h_dm vs w-sigma mismatch\n(time-dependent NN -- should RECOVER)')
    ax1.legend(fontsize=8); ax1.grid(alpha=0.3); ax1.invert_xaxis()

    ax2.plot(w0_sf_arr, A_nn_td_median, 'C2D-', lw=2, ms=7, label='Time-dep NN+h (median)')
    ax2.fill_between(w0_sf_arr, A_nn_td_lo, A_nn_td_hi, color='C2', alpha=0.2, label='16-84%')
    ax2.set_xlabel('Fitting $\\sigma_w$  (true = 20)')
    ax2.set_ylabel('$A_{\\rm nn}$')
    ax2.set_title('Test 2: NN amplitude vs w-sigma mismatch\n(A_nn grows to absorb systematic)')
    ax2.legend(fontsize=8); ax2.grid(alpha=0.3); ax2.invert_xaxis()

    fig.suptitle('Test 2: Mismatched $\\sigma_w$ (narrower) + Time-Dep NN  (should recover $h_{\\rm dm}$)',
                 fontsize=13, y=1.03)
    plt.tight_layout()
    plt.savefig(f'{save_dir}/test2_sigma_mismatch_timedep_nn.png', dpi=150, bbox_inches='tight')
    plt.show()

    # --- Plot 2: combined (h-only vs static NN vs time-dep NN) ---
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5), dpi=140)
    ax1.axhline(h_true, color='k', ls='--', lw=1.5, label=f'True $h$ = {h_true}')
    ax1.plot(w0_sf_arr, h_only_arr, 'C0o-', lw=2, ms=7, label='h-only')
    ax1.plot(w0_sf_arr, h_nn_median,    'C1s-', lw=2, ms=7, label='Static NN+h (median)')
    ax1.fill_between(w0_sf_arr, h_nn_lo, h_nn_hi, color='C1', alpha=0.15)
    ax1.plot(w0_sf_arr, h_nn_td_median, 'C2D-', lw=2, ms=7, label='Time-dep NN+h (median)')
    ax1.fill_between(w0_sf_arr, h_nn_td_lo, h_nn_td_hi, color='C2', alpha=0.15)
    ax1.set_xlabel('Fitting $\\sigma_w$  (true = 20)')
    ax1.set_ylabel('$h_{\\rm pc,dm}$ fitted [pc]')
    ax1.set_title('$h_{\\rm dm}$ recovery: all 3 methods')
    ax1.legend(fontsize=8); ax1.grid(alpha=0.3); ax1.invert_xaxis()

    ax2.plot(w0_sf_arr, A_nn_median,    'C1s-', lw=2, ms=7, label='Static NN (median)')
    ax2.fill_between(w0_sf_arr, A_nn_lo, A_nn_hi, color='C1', alpha=0.15)
    ax2.plot(w0_sf_arr, A_nn_td_median, 'C2D-', lw=2, ms=7, label='Time-dep NN (median)')
    ax2.fill_between(w0_sf_arr, A_nn_td_lo, A_nn_td_hi, color='C2', alpha=0.15)
    ax2.set_xlabel('Fitting $\\sigma_w$  (true = 20)')
    ax2.set_ylabel('$A_{\\rm nn}$')
    ax2.set_title('NN amplitude: static vs time-dependent')
    ax2.legend(fontsize=8); ax2.grid(alpha=0.3); ax2.invert_xaxis()

    fig.suptitle('Combined: h-only vs Static NN vs Time-Dep NN under $\\sigma_w$ mismatch (narrower)',
                 fontsize=13, y=1.03)
    plt.tight_layout()
    plt.savefig(f'{save_dir}/combined_mismatch_comparison.png', dpi=150, bbox_inches='tight')
    plt.show()

    # --- Summary table ---
    print(f"\n{'='*80}")
    print(f"  Combined Summary: h_dm fitted vs w-sigma mismatch")
    print(f"  True h_dm = {h_true:.0f} pc,  true z0_sigma = {z0_sigma}, w0_sigma = {w0_sigma}")
    print(f"{'='*80}")
    print(f"  {'fit_w_sig':>10s}  {'ratio':>8s}  {'h-only':>10s}  {'Static NN':>10s}  {'TimeDep NN':>10s}  {'A_static':>10s}  {'A_timedep':>10s}")
    print(f"  {'-'*10}  {'-'*8}  {'-'*10}  {'-'*10}  {'-'*10}  {'-'*10}  {'-'*10}")
    for i, s in enumerate(w0_sf_arr):
        ratio = s / w0_sigma
        print(f"  {s:10.1f}  {ratio:6.2f}x  {h_only_arr[i]:10.2f}  {h_nn_median[i]:10.2f}  "
              f"{h_nn_td_median[i]:10.2f}  {A_nn_median[i]:10.2e}  {A_nn_td_median[i]:10.2e}")

    return {
        "test2_nn_results": test2_nn_results,
        "h_nn_td_all": h_nn_td_all, "A_nn_td_all": A_nn_td_all,
        "h_nn_td_median": h_nn_td_median, "h_nn_td_lo": h_nn_td_lo, "h_nn_td_hi": h_nn_td_hi,
        "A_nn_td_median": A_nn_td_median, "A_nn_td_lo": A_nn_td_lo, "A_nn_td_hi": A_nn_td_hi,
    }
