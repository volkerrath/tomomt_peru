"""
Berryman (1980) self-consistent (SC) effective medium model
================================================================

Faithful implementation of BOTH papers in the series:
  Part I:  Berryman, J. G. (1980), "Long-wavelength propagation in
           composite elastic media I. Spherical inclusions,"
           J. Acoust. Soc. Am., 68(6), 1809-1819.
  Part II: Berryman, J. G. (1980), "Long-wavelength propagation in
           composite elastic media II. Ellipsoidal inclusions,"
           J. Acoust. Soc. Am., 68(6), 1820-1831.

Equation numbers below match the respective paper (noted per function).

How this differs from Kuster-Toksoz (kt_dem_model.kuster_toksoz)
--------------------------------------------------------------------
KT embeds each inclusion in the ORIGINAL matrix and is explicit
(closed-form), asymmetric under exchange of matrix/inclusion labels,
and for SPHERICAL inclusions is IDENTICAL to the Hashin-Shtrikman
bounds. For NONSPHERICAL inclusions, Part II shows KT can actually
VIOLATE the Hashin-Shtrikman bounds at high enough concentration
(most severely for disks, where KT violates the bounds at essentially
any finite concentration, since its "noninteraction assumption"
requires concentration <~ aspect ratio) - i.e. KT is not just
approximate but can be provably wrong outside a narrow validity range.

SC instead embeds every constituent (matrix AND inclusions alike) in
the unknown EFFECTIVE medium itself, and solves (iteratively, since
the equations are implicit) for the medium that makes the net
scattering vanish - "impedance matching." SC is:
  - symmetric under exchange of constituent labels,
  - PROVABLY always between the Hashin-Shtrikman bounds for spherical
    inclusions (Part I), and for ellipsoidal inclusions ALSO satisfies
    the tighter Miller bounds (Part II) - checked numerically there
    for a wide range of concentrations and shapes,
  - able to reproduce a threshold-of-rigidity / percolation-like
    transition (e.g. a solid-in-fluid suspension has zero effective
    shear modulus below some solid concentration, then rises steeply)
    that KT cannot reproduce at all. For ellipsoidal inclusions the
    threshold location shifts with aspect ratio (Part II, Section IV):
    from c1=0.4 for spheres, continuously down to c1=0 for needles and
    disks/cracks (no threshold at all in those limits).
  - numerically trickier: implicit, iterative, and has a genuine
    numerical singularity at that threshold concentration, whose exact
    location depends on inclusion shape and is not predictable in
    closed form (Part II states this explicitly - "no method of
    predicting the location of the singular point ... has yet been
    found"). This module uses damped (relaxed) iteration and the
    papers' Appendix A / Eq.(50) geometric-mean ansatz as an initial
    guess to help convergence near that regime, but convergence is not
    guaranteed arbitrarily close to the singular point - if iteration
    fails to converge, that is itself informative (you are likely near
    the singular point) rather than a bug to silently paper over.

Part I: spherical inclusions (exact)
----------------------------------------
Auxiliary functionals (Eqs. 37, 38):
    K(mu)  = [ sum_i c_i / (K_i + 4/3 mu) ]^-1 - 4/3 mu
    mu(F)  = [ sum_i c_i / (mu_i + F)     ]^-1 - F

F as a function of the CURRENT effective medium (Eq. 43):
    F(K, mu) = (mu/6) * (9K + 8mu) / (K + 2mu)

Self-consistent equations (Eqs. 28, 30a, 29):
    1/(K* + 4/3 mu*) = sum_i c_i / (K_i + 4/3 mu*)      i.e. K* = K(mu*)
    1/(mu* + F*)     = sum_i c_i / (mu_i + F*)          i.e. mu* = mu(F*), F* = F(K*, mu*)
    rho* = sum_i c_i rho_i                               (Eq. 29, no coupling - exact)

Solved by fixed-point iteration:
    mu* -> K* = K(mu*) -> F* = F(K*, mu*) -> mu*_new = mu(F*) -> repeat.

Part II: ellipsoidal inclusions (exact)
----------------------------------------
The fully self-consistent formulas (Eqs. 46, 47) generalize Part I to
inclusions of arbitrary aspect ratio, using the SAME P, Q geometric
factors as Kuster-Toksoz (Berryman's own Appendix, Eqs. A3-A18 - these
are exactly the formulas already implemented in
kt_dem_model._PQ_general, confirmed by direct term-by-term comparison;
see that module) but with each P*i, Q*i evaluated using the CURRENT
effective medium (K*, mu*) as the matrix, rather than a fixed host:
    sum_i c_i P*i (K_i - K*) = 0
    sum_i c_i Q*i (mu_i - mu*) = 0
solved by the paper's own practical iteration scheme (Eqs. 32, 33):
    (K*)_{n+1}  = sum_i c_i K_i  (P*i)_n / sum_i c_i (P*i)_n
    (mu*)_{n+1} = sum_i c_i mu_i (Q*i)_n / sum_i c_i (Q*i)_n
where (P*i)_n, (Q*i)_n are evaluated using (K*)_n, (mu*)_n as the
matrix. Reduces exactly to the Part I spherical formulas when all
aspect ratios equal 1.
"""

import numpy as np
from kt_dem_model import _PQ_general


def _K_of_mu(mu, K_list, c_list):
    """Eq. (37): K(mu)."""
    s = sum(c / (K_i + 4.0 / 3.0 * mu) for c, K_i in zip(c_list, K_list))
    return 1.0 / s - 4.0 / 3.0 * mu


def _mu_of_F(F, mu_list, c_list):
    """Eq. (38): mu(F)."""
    s = sum(c / (mu_i + F) for c, mu_i in zip(c_list, mu_list))
    return 1.0 / s - F


def _F_of_K_mu(K, mu):
    """Eq. (43): F as a function of the current effective medium."""
    return (mu / 6.0) * (9 * K + 8 * mu) / (K + 2 * mu)


def _initial_guess(K_list, mu_list, c_list):
    """
    Starting guess for mu*. Uses the simple concentration-weighted
    (Voigt-like) average of mu_i as a default; if any constituent has
    mu_i == 0 (a fluid phase) or mu_i is orders of magnitude smaller
    than the others, uses the paper's Appendix A geometric-mean ansatz
    (Eq. A8) instead, since that is specifically recommended there as
    a good estimate near the singular point.
    """
    mu_arr = np.asarray(mu_list, dtype=float)
    c_arr = np.asarray(c_list, dtype=float)

    if np.any(mu_arr <= 0) or (mu_arr.max() > 0 and mu_arr.min() / mu_arr.max() < 1e-6):
        # Appendix A, Eq. (A8)-style geometric-mean ansatz, generalized
        # to n phases via the harmonic/arithmetic mean combination used
        # in Eq. (A7)-(A8): mu* ~ sqrt( <mu> * <1/mu>^-1 )
        mu_safe = np.where(mu_arr <= 0, 1e-6 * max(mu_arr.max(), 1.0), mu_arr)
        mean_mu = np.sum(c_arr * mu_safe)
        mean_inv_mu = np.sum(c_arr / mu_safe)
        harmonic = 1.0 / mean_inv_mu
        return np.sqrt(mean_mu * harmonic)
    else:
        return np.sum(c_arr * mu_arr)  # Voigt average


def self_consistent(K_list, mu_list, c_list, rho_list=None,
                     max_iter=500, tol=1e-8, relax=0.5, return_info=False):
    """
    Part I: Berryman (1980) self-consistent effective moduli for an
    n-phase composite of SPHERICAL inclusions.

    Parameters
    ----------
    K_list, mu_list : sequence of float
        Bulk and shear moduli of each constituent phase [Pa].
    c_list : sequence of float
        Volume concentrations of each phase (must sum to 1).
    rho_list : sequence of float, optional
        Densities of each phase [kg/m3]. If given, effective density
        is also returned (Eq. 29 - exact, no iteration).
    max_iter : int
        Maximum number of fixed-point iterations.
    tol : float
        Convergence tolerance on the relative change in mu*.
    relax : float, default 0.5
        Under-relaxation factor (0 < relax <= 1) applied to each mu*
        update: mu_new_relaxed = relax*mu_new + (1-relax)*mu_old. The
        unrelaxed iteration (relax=1) can oscillate or diverge,
        especially near the singular point discussed above; damping
        trades convergence speed for robustness.
    return_info : bool
        If True, also return a dict with 'converged' (bool) and
        'n_iter' (int) for diagnostics.

    Returns
    -------
    Kstar, mustar[, rhostar][, info]
    """
    K_list = list(K_list)
    mu_list = list(mu_list)
    c_list = list(c_list)
    if not np.isclose(sum(c_list), 1.0, atol=1e-6):
        raise ValueError(f"Concentrations must sum to 1 (got {sum(c_list)})")

    mu_star = _initial_guess(K_list, mu_list, c_list)
    converged = False
    n_iter = 0
    for n_iter in range(1, max_iter + 1):
        K_star = _K_of_mu(mu_star, K_list, c_list)
        F_star = _F_of_K_mu(K_star, mu_star)
        mu_new = _mu_of_F(F_star, mu_list, c_list)
        mu_new = max(mu_new, 0.0)  # shear modulus is non-negative

        mu_relaxed = relax * mu_new + (1 - relax) * mu_star
        denom = max(abs(mu_star), 1e-12)
        if abs(mu_relaxed - mu_star) / denom < tol:
            mu_star = mu_relaxed
            converged = True
            break
        mu_star = mu_relaxed

    K_star = _K_of_mu(mu_star, K_list, c_list)

    result = [K_star, mu_star]
    if rho_list is not None:
        rho_star = sum(c * r for c, r in zip(c_list, rho_list))  # Eq. (29)
        result.append(rho_star)
    if return_info:
        result.append({'converged': converged, 'n_iter': n_iter})
    return tuple(result) if len(result) > 1 else result[0]


def self_consistent_two_phase(K1, mu1, rho1, K2, mu2, rho2, c1, **kwargs):
    """
    Convenience wrapper for the common two-phase SPHERICAL case (e.g.
    matrix + single inclusion type). c1 is the volume concentration of
    phase 1.
    """
    return self_consistent([K1, K2], [mu1, mu2], [c1, 1 - c1],
                            rho_list=[rho1, rho2], **kwargs)


def self_consistent_ellipsoidal(K_list, mu_list, alpha_list, c_list, rho_list=None,
                                 max_iter=500, tol=1e-8, relax=0.5, return_info=False):
    """
    Part II: Berryman (1980) self-consistent effective moduli for an
    n-phase composite of ELLIPSOIDAL (spheroidal) inclusions of
    arbitrary aspect ratio, generalizing self_consistent() above.

    Uses the same aspect-ratio convention as kt_dem_model.py:
    alpha = 1 is a sphere; alpha < 1 is oblate (disk-like as alpha -> 0);
    alpha > 1 is prolate (needle-like as alpha -> inf).

    Parameters
    ----------
    K_list, mu_list : sequence of float
        Bulk and shear moduli of each constituent phase [Pa].
    alpha_list : sequence of float
        Aspect ratio of each phase's inclusion shape (every constituent
        needs an entry - self-consistent embedding treats all phases
        symmetrically, so there is no single fixed "matrix shape").
    c_list : sequence of float
        Volume concentrations of each phase (must sum to 1).
    rho_list, max_iter, tol, relax, return_info : as in self_consistent().

    Returns
    -------
    Kstar, mustar[, rhostar][, info]

    Notes
    -----
    Reduces exactly to self_consistent() when all entries of
    alpha_list equal 1.0 (spheres) - useful as a self-check.
    """
    K_list = list(K_list)
    mu_list = list(mu_list)
    alpha_list = list(alpha_list)
    c_list = list(c_list)
    if not np.isclose(sum(c_list), 1.0, atol=1e-6):
        raise ValueError(f"Concentrations must sum to 1 (got {sum(c_list)})")

    mu_star = _initial_guess(K_list, mu_list, c_list)
    K_star = sum(c * K_i for c, K_i in zip(c_list, K_list))  # Voigt initial guess

    converged = False
    n_iter = 0
    for n_iter in range(1, max_iter + 1):
        P_list, Q_list = [], []
        for K_i, mu_i, alpha_i in zip(K_list, mu_list, alpha_list):
            P, Q = _PQ_general(K_star, mu_star, K_i, mu_i, alpha_i)
            P_list.append(P)
            Q_list.append(Q)

        den_K = sum(c * P for c, P in zip(c_list, P_list))
        K_new = sum(c * K_i * P for c, K_i, P in zip(c_list, K_list, P_list)) / den_K

        den_mu = sum(c * Q for c, Q in zip(c_list, Q_list))
        mu_new = sum(c * mu_i * Q for c, mu_i, Q in zip(c_list, mu_list, Q_list)) / den_mu
        mu_new = max(mu_new, 0.0)

        K_relaxed = relax * K_new + (1 - relax) * K_star
        mu_relaxed = relax * mu_new + (1 - relax) * mu_star

        denom_K = max(abs(K_star), 1e-12)
        denom_mu = max(abs(mu_star), 1e-12)
        if (abs(K_relaxed - K_star) / denom_K < tol and
                abs(mu_relaxed - mu_star) / denom_mu < tol):
            K_star, mu_star = K_relaxed, mu_relaxed
            converged = True
            break
        K_star, mu_star = K_relaxed, mu_relaxed

    result = [K_star, mu_star]
    if rho_list is not None:
        rho_star = sum(c * r for c, r in zip(c_list, rho_list))  # exact, uncoupled
        result.append(rho_star)
    if return_info:
        result.append({'converged': converged, 'n_iter': n_iter})
    return tuple(result) if len(result) > 1 else result[0]


def self_consistent_ellipsoidal_two_phase(K1, mu1, alpha1, rho1,
                                           K2, mu2, alpha2, rho2, c1, **kwargs):
    """
    Convenience wrapper for the common two-phase ELLIPSOIDAL case.
    c1 is the volume concentration of phase 1. alpha1, alpha2 are each
    phase's own inclusion aspect ratio (see self_consistent_ellipsoidal).
    """
    return self_consistent_ellipsoidal(
        [K1, K2], [mu1, mu2], [alpha1, alpha2], [c1, 1 - c1],
        rho_list=[rho1, rho2], **kwargs)


# ---------------------------------------------------------------
# Examples / validation
# ---------------------------------------------------------------
if __name__ == "__main__":
    from kt_dem_model import kuster_toksoz

    # --- Part I validation: solid + water, spherical inclusions ---
    K1, mu1, rho1 = 0.44e11, 0.37e11, 2700.0
    K2, mu2, rho2 = 0.022e11, 1e-6, 1000.0
    
    ZERO = 1e-12
    rho1 = 2.84     # g/cm^3
    c11 = 136.03   # GPa
    c12 = 56.31    # GPa
    c44 = 39.86    # GPa
    K1 = (c11+2*c12)/3.0
    G1 = c44
    mu1 = G1
    
    # Melt phase properties
    rho2 = 2.7     # g/cm^3
    K2 = 1.491e1   # GPa
    G2 = ZERO      # GPa
    mu2 = G2




    print("=== Part I: spherical inclusions (solid + water) ===")
    print(f"{'c1':>6} {'K_SC(GPa)':>10} {'mu_SC(GPa)':>11} {'converged':>10} "
          f"| {'K_KT(GPa)':>10} {'mu_KT(GPa)':>11}")
    for c1 in [0.0, 0.2, 0.35, 0.38, 0.4, 0.42, 0.45, 0.6, 0.8, 1.0]:
        c1_eff = min(max(c1, 1e-6), 1 - 1e-6)
        K_sc, mu_sc, rho_sc, info = self_consistent_two_phase(
            K1, mu1, rho1, K2, mu2, rho2, c1_eff, return_info=True)
        inclusions = [{'K': K1, 'mu': mu1, 'x': c1_eff, 'alpha': 1.0}]
        K_kt, mu_kt = kuster_toksoz(K2, mu2, inclusions)
        print(f"{c1:6.2f} {K_sc/1e9:10.3f} {mu_sc/1e9:11.4f} {str(info['converged']):>10} "
              f"| {K_kt/1e9:10.3f} {mu_kt/1e9:11.4f}")




    # --- Part II validation: reproduce the paper's own reported
    # singular-point locations for needle/disk-like inclusions at
    # aspect ratio 0.1 (paper's Section IV / Figs. 2-3 captions):
    #   prolate spheroids, aspect ratio 0.1 -> singular point c1 ~ 0.24
    #   oblate  spheroids, aspect ratio 0.1 -> singular point c1 ~ 0.33
    # Paper's aspect ratio convention is always <=1 (c/a); this
    # module's convention (matching kt_dem_model.py) uses alpha>1 for
    # prolate (= a/c, the reciprocal) and alpha<1 for oblate (= c/a,
    # matches directly). So paper's "prolate, aspect ratio 0.1" is
    # alpha=10 here; "oblate, aspect ratio 0.1" is alpha=0.1 here.
    print("\n=== Part II: locating the singular point vs. inclusion shape ===")
    print("(paper reports c1~0.24 for prolate @ shape 0.1, c1~0.33 for oblate @ shape 0.1)")
    # NOTE: the singular-point/threshold behavior is reported for the
    # paper's Figs. 5-6 (velocity/attenuation) example, which uses the
    # WATER constituent (mu2 ~ 0) - NOT the stiffer solid-solid pair
    # (K2=0.14 Mb, mu2=0.10 Mb) used for the Hashin-Shtrikman/Miller
    # bound comparisons in Figs. 1-4. Using the wrong pair here would
    # give no threshold at all (both phases solid -> no percolation).
    K1b, mu1b, rho1b = 0.44e11, 0.37e11, 2700.0
    K2b, mu2b, rho2b = 0.022e11, 1e-6, 1000.0   # water, same as Part I / Figs. 5-6

    for label, alpha_incl in [("prolate (paper aspect ratio 0.1 -> alpha=10 here)", 10.0),
                               ("oblate  (paper aspect ratio 0.1 -> alpha=0.1 here)", 0.1)]:
        print(f"\n{label}:")
        print(f"{'c1':>6} {'mu_SC(GPa)':>11} {'converged':>10}")
        for c1 in np.arange(0.05, 0.55, 0.05):
            # phase 1 = solid, whose grain shape is what the paper varies
            # (prolate/oblate); phase 2 = water, left spherical (alpha=1)
            K_sc, mu_sc, rho_sc, info = self_consistent_ellipsoidal_two_phase(
                K1b, mu1b, alpha_incl, rho1b,
                K2b, mu2b, 1.0, rho2b,
                c1=c1, return_info=True)
            print(f"{c1:6.2f} {mu_sc/1e9:11.4f} {str(info['converged']):>10}")
