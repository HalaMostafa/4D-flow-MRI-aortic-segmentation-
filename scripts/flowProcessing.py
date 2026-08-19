# -*- coding: utf-8 -*-
# -----------------------------------------------------------------------------
# RACLETTE · Realistic Synthetic Aortic 4D/5D Flow MRI
# Description : Velocity extraction, IVSD estimation, k-space noise addition,
#               and Bayesian multi-venc unfolding for 4D Flow MRI signals
# Version     : 1.0
# Authors     : P. Dirix; S. Buoso; S. Kozerke
# License     : MIT (code) · CC-BY-NC-4.0 (data)
# Repo        : https://gitlab.ethz.ch/ibt-cmr/publications/raclette
# Dataset     : https://doi.org/10.3929/ethz-c-000799752
# Cite        : https://doi.org/10.1007/978-3-031-73281-2_15 (paper)
#               https://doi.org/10.3929/ethz-c-000799752 (dataset)
# -----------------------------------------------------------------------------
import os

import numpy as np
from scipy.fft import fftn, ifftn

try:
    import numexpr as ne
    _NUMEXPR = True
except ImportError:
    _NUMEXPR = False

try:
    from numba import jit, prange, set_num_threads
except ImportError:
    prange = range

    def set_num_threads(n):
        pass

    def jit(*args, **kwargs):
        if len(args) == 1 and callable(args[0]) and not kwargs:
            return args[0]
        return lambda func: func


def extract_velocities(signal, kvs):
    """Phase-contrast velocity from complex MRI signal.

    Parameters
    ----------
    signal : (*dims, n_enc, ...)  complex  – reference encoding is index 0 along axis 3
    kvs    : (n_enc,)             float    – velocity-encoding sensitivities [rad/(m/s)]

    Returns
    -------
    np.ndarray, shape (*dims, n_enc-1, ...), float64  [m/s]
    """
    a = signal[:, :, :, 1:, ...]
    b = signal[:, :, :, 0:1, ...]
    kv_shape = [1] * a.ndim
    kv_shape[3] = -1
    kv = kvs[1:].reshape(kv_shape)
    if _NUMEXPR:
        return np.angle(ne.evaluate('a * conj(b)')) / kv
    else:
        return np.angle(a * np.conj(b)) / kv


def extract_ivsds(signal, kvs):
    """Intravoxel velocity standard deviation (IVSD) from complex MRI signal.

    Parameters
    ----------
    signal : (*dims, n_enc, ...)  complex  – reference encoding is index 0 along axis 3
    kvs    : (n_enc,)             float    – velocity-encoding sensitivities [rad/(m/s)]

    Returns
    -------
    np.ndarray, shape (*dims, n_enc-1, ...), float64  [m/s]
    """
    abs_ref  = np.abs(signal[:, :, :, :1, ...])
    abs_rest = np.abs(signal[:, :, :, 1:, ...])
    kv_shape = [1] * signal.ndim
    kv_shape[3] = -1
    kv = kvs[1:].reshape(kv_shape)
    if _NUMEXPR:
        with np.errstate(divide='ignore', invalid='ignore'):
            result = ne.evaluate('where(abs_rest > 0, sqrt((2 / kv**2) * log(abs_ref / abs_rest)), 0)')
        return np.nan_to_num(result)
    else:
        with np.errstate(divide='ignore', invalid='ignore'):
            log_ratio = np.log(abs_ref / abs_rest)
        return np.nan_to_num(np.sqrt((2 / kv**2) * log_ratio))


def add_kspace_noise(signal, mask, snr=20, axes=(0, 1, 2)):
    """Add complex Gaussian k-space noise to a 4D-flow signal (SNR simulation).

    Parameters
    ----------
    signal : (*dims, n_enc, n_frames)  complex
    mask   : (*dims, n_frames)         float/bool – used to estimate the signal level
    snr    : float                     target signal-to-noise ratio (reference
                                        encoding magnitude / noise std)
    axes   : tuple of int              spatial axes over which the FFT roundtrip is
                                        performed (default: the 3 spatial axes)

    Returns
    -------
    np.ndarray, same shape/dtype as `signal` – noisy signal
    """
    mean_S = np.mean(np.abs(signal[..., 0, :])[np.abs(mask) > 0.5])
    sigma = mean_S / snr
    noise = (np.random.normal(0, sigma, signal.shape)
             + 1j * np.random.normal(0, sigma, signal.shape))
    k = fftn(signal, axes=axes, norm='ortho', workers=-1)
    return ifftn(k + noise, axes=axes, norm='ortho', workers=-1)


@jit(nopython=True, parallel=True, cache=True)
def _bayes_solve_grid_multires(sdata_all, kv, kv_sq):
    """Per-voxel (velocity, IVSD) MAP estimate via coarse-to-fine grid search.

    Numerically efficient grid-search reformulation of the Bayesian unfolding
    MAP estimate (originally a per-voxel Nelder-Mead optimisation, see
    `bayes_unfold`), from
    https://gitlab.ethz.ch/pdirix/Synthetic4DFlowMRI_Tutorial.

    Parameters
    ----------
    sdata_all : (N_voxels, N_enc) complex64 – img * conj(img[:, 0]) per voxel
    kv        : (N_enc,)         float32   – velocity-encoding sensitivities,
                kv[0] == 0 (reference)
    kv_sq     : (N_enc,)         float32   – kv**2

    Returns
    -------
    velocity : (N_voxels,) float32 – sign-matched to `extract_velocities`
    ivsd     : (N_voxels,) float32 – sigma (not sigma^2), matching `extract_ivsds`
    """
    n_voxels = sdata_all.shape[0]
    results_v = np.zeros(n_voxels, dtype=np.float32)
    results_std = np.zeros(n_voxels, dtype=np.float32)

    N_enc = kv.shape[0]

    Nv, Ns = 21, 21
    Nvf, Nsf = 21, 21
    delta_v_coarse = 1e-1
    delta_v_fine = 2 * delta_v_coarse / Nv
    delta_s_coarse = 0.15
    delta_s_fine = 2 * delta_s_coarse / Ns

    for n in prange(n_voxels):
        sdata = sdata_all[n]
        sdata_r = sdata.real
        sdata_i = sdata.imag
        denom = np.sum(sdata_r ** 2 + sdata_i ** 2)
        if denom == 0.0:
            denom = 1e-10

        # Initial velocity estimate from phase difference between first and last encoding
        angle0 = np.angle(sdata[0])
        angle2 = np.angle(sdata[-1])
        v_est = -np.angle(np.exp(1j * (angle2 - angle0))) / kv[-1]

        # Initial turbulence estimate from log-ratio of magnitudes
        s0 = np.abs(sdata[0])
        s1 = np.abs(sdata[1])
        s_est = np.sqrt((2.0 / (kv[1] ** 2)) * np.log(s0 / s1)) if s1 > 0 else 0.001
        if s_est < 0.001 or np.isnan(s_est):
            s_est = 0.001

        # Coarse velocity and sigma grids
        v_grid = np.linspace(v_est - delta_v_coarse, v_est + delta_v_coarse, Nv).astype(np.float32)
        sigma_grid = np.linspace(max(0.001, s_est - delta_s_coarse), s_est + delta_s_coarse, Ns).astype(np.float32)

        # Precompute sin and cos of v*k for each point in v_grid
        vk_grid = np.empty((Nv, N_enc), dtype=np.float32)
        sin_vk_grid = np.empty((Nv, N_enc), dtype=np.float32)
        cos_vk_grid = np.empty((Nv, N_enc), dtype=np.float32)
        for i in range(Nv):
            vk = v_grid[i] * kv
            vk_grid[i, :] = vk
            sin_vk_grid[i, :] = np.sin(vk)
            cos_vk_grid[i, :] = np.cos(vk)

        # Precompute exponential signal decay for each sigma
        exp_half_grid = np.empty((Ns, N_enc), dtype=np.float32)
        exp_eig_grid = np.empty(Ns, dtype=np.float32)
        for j in range(Ns):
            sig = sigma_grid[j]
            if sig < 0.001:
                sig = 0.001
            sigsq = sig ** 2
            exp_eig = np.exp(-sigsq * kv_sq)
            exp_eig_grid[j] = 1.0 / np.sum(exp_eig)
            exp_half_grid[j, :] = np.exp(-0.5 * sigsq * kv_sq)

        # Grid search to find minimum posterior
        min_post = 1e10
        best_v = 0.0
        best_s = 1.0
        for i in range(Nv):
            for j in range(Ns):
                eig1 = exp_eig_grid[j]
                exp_half = exp_half_grid[j, :]
                sin_vk = sin_vk_grid[i, :]
                cos_vk = cos_vk_grid[i, :]

                # Compute projection onto model basis
                h1 = np.sum(exp_half * (sdata_r * sin_vk + sdata_i * cos_vk))
                h2 = np.sum(exp_half * (sdata_r * cos_vk - sdata_i * sin_vk))

                # Normalize projections
                h1 *= np.sqrt(eig1)
                h2 *= np.sqrt(eig1)

                mh2 = h1 * h1 + h2 * h2
                frac = 1 - mh2 / denom
                if frac <= 0.0:
                    continue

                # Negative log-posterior
                post = -np.log(eig1 * frac ** ((2 - 2 * N_enc) / 2))
                if post < min_post:
                    min_post = post
                    best_v = v_grid[i]
                    best_s = sigma_grid[j]

        # Fine search around best coarse estimate
        v_fine_adj = np.linspace(best_v - delta_v_fine, best_v + delta_v_fine, Nvf).astype(np.float32)
        s_fine_adj = np.linspace(max(0.001, best_s - delta_s_fine), best_s + delta_s_fine, Nsf).astype(np.float32)
        min_post_f = 1e10
        best_vf = best_v
        best_sf = best_s

        # Precompute v * k
        vk_fine = np.empty((len(v_fine_adj), len(kv)), dtype=np.float32)
        for i in range(len(v_fine_adj)):
            for k in range(len(kv)):
                vk_fine[i, k] = v_fine_adj[i] * kv[k]

        # Loop through fine grid
        for i in range(len(v_fine_adj)):
            for j in range(len(s_fine_adj)):
                v = v_fine_adj[i]
                sig = s_fine_adj[j]
                if sig < 0.01:
                    sig = 0.01
                sigsq = sig * sig

                # Compute model response
                exp_eig = np.exp(-sigsq * kv_sq)
                eig1 = 1.0 / np.sum(exp_eig)
                exp_half = np.exp(-0.5 * sigsq * kv_sq)
                sin_vk = np.sin(vk_fine[i])
                cos_vk = np.cos(vk_fine[i])

                # Projection again
                h1 = np.sum(exp_half * (sdata_r * sin_vk + sdata_i * cos_vk))
                h2 = np.sum(exp_half * (sdata_r * cos_vk - sdata_i * sin_vk))

                h1 *= np.sqrt(eig1)
                h2 *= np.sqrt(eig1)

                mh2 = h1 * h1 + h2 * h2
                frac = 1 - mh2 / denom
                if frac <= 0.0:
                    continue

                post = -np.log(eig1 * frac ** ((2 - 2 * N_enc) / 2))
                if post < min_post_f:
                    min_post_f = post
                    best_vf = v
                    best_sf = sig

        # Store final estimates
        results_v[n] = best_vf
        results_std[n] = best_sf

    return -results_v, results_std  # flip sign per convention


def bayes_unfold(signal_triplet, kv, mask, parallel=True):
    """Bayesian dual-point velocity/IVSD unfolding for one encoding direction.

    Coarse-to-fine grid search MAP estimate of (velocity, IVSD) for each masked
    voxel. The original (Nelder-Mead) formulation of this Bayesian unfolding is
    from https://gitlab.ethz.ch/ibt-cmr/publications/4dflowmrisynthesis
    (https://doi.org/10.1038/s41598-022-20121-x); this is the numerically
    efficient grid-search reformulation from
    https://gitlab.ethz.ch/pdirix/Synthetic4DFlowMRI_Tutorial.

    Parameters
    ----------
    signal_triplet : (*dims, 3)  complex – [reference, low-venc point, high-venc
                      point] for ONE encoding direction
    kv              : (3,)       float   – [0, kv_lo, kv_hi]
    mask            : (*dims)    bool
    parallel        : bool                 use all CPU cores (default: True)

    Returns
    -------
    velocity : (*dims) float  [m/s] – sign-matched to `extract_velocities`
    ivsd     : (*dims) float  [m/s] – sigma (not sigma^2), matching `extract_ivsds`
    """
    shape = signal_triplet.shape[:-1]
    flat = signal_triplet.reshape(-1, signal_triplet.shape[-1])
    mask_flat = mask.reshape(-1) > 0
    kv = np.asarray(kv, dtype=np.float32)
    kv_sq = kv * kv

    img = flat[mask_flat]
    sdata = (img * np.conj(img[:, 0:1])).astype(np.complex64)

    set_num_threads(os.cpu_count() if parallel else 1)
    results_v, results_std = _bayes_solve_grid_multires(sdata, kv, kv_sq)

    velocity = np.zeros(flat.shape[0])
    ivsd = np.zeros(flat.shape[0])
    velocity[mask_flat] = results_v
    ivsd[mask_flat] = results_std

    return velocity.reshape(shape), ivsd.reshape(shape)


def get_cartesian_velocities(v_dir, enc_dirs):
    """Reconstruct Cartesian velocities from multi-directional encodings.

    Pseudo-inverse reconstruction from
    https://gitlab.ethz.ch/pdirix/Synthetic4DFlowMRI_Tutorial, generalised to
    broadcast over arbitrary leading dims via `@` instead of an explicit
    flatten/reshape.

    Parameters
    ----------
    v_dir    : (..., Ndir)  float – velocities along each encoding direction [m/s]
    enc_dirs : (Ndir, 3)    float – encoding direction vectors (need not be unit)

    Returns
    -------
    np.ndarray, shape (..., 3), float – Cartesian velocities (vx, vy, vz) [m/s]
    """
    A_dir = enc_dirs / np.linalg.norm(enc_dirs, axis=1, keepdims=True)
    Ainv = np.linalg.pinv(A_dir)
    return v_dir @ Ainv.T


def calc_rst_from_ivsd(ivsd_dir, enc_dirs, density=1060):
    """Reynolds Stress Tensor from multi-directional IVSDs.

    Pseudo-inverse reconstruction from
    https://gitlab.ethz.ch/pdirix/Synthetic4DFlowMRI_Tutorial, generalised
    (via `np.stack`/`@`) from a 2D `(Nvox, Nenc)`-only input to arbitrary
    leading dims `(..., Ndir)`.

    Parameters
    ----------
    ivsd_dir : (..., Ndir)  float – intra-voxel standard deviations (sigma, not
               sigma^2) along each encoding direction [m/s]
    enc_dirs : (Ndir, 3)    float – encoding direction vectors
    density  : float                fluid density [kg/m^3]

    Returns
    -------
    np.ndarray, shape (..., 6), float – RST = density * [Rxx,Ryy,Rzz,Rxy,Rxz,Ryz] [Pa]
    """
    H = np.stack([
        enc_dirs[:, 0] ** 2,
        enc_dirs[:, 1] ** 2,
        enc_dirs[:, 2] ** 2,
        2 * enc_dirs[:, 0] * enc_dirs[:, 1],
        2 * enc_dirs[:, 0] * enc_dirs[:, 2],
        2 * enc_dirs[:, 1] * enc_dirs[:, 2],
    ], axis=1)
    kv_magsq = np.sum(enc_dirs ** 2, axis=1, keepdims=True)
    Hinv = np.linalg.pinv(H / kv_magsq)

    return (ivsd_dir ** 2) @ Hinv.T * density
