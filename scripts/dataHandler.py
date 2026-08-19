# -*- coding: utf-8 -*-
# -----------------------------------------------------------------------------
# RACLETTE · Realistic Synthetic Aortic 4D/5D Flow MRI
# Description : I/O library for loading and saving RACLETTE .pv / HDF5 /
#               NIfTI / NRRD / DICOM / VTK 4D Flow MRI frames and field data
# Version     : 1.0
# Authors     : P. Dirix; S. Buoso; S. Kozerke
# License     : MIT (code) · CC-BY-NC-4.0 (data)
# Repo        : https://gitlab.ethz.ch/ibt-cmr/publications/raclette
# Dataset     : https://doi.org/10.3929/ethz-c-000799752
# Cite        : https://doi.org/10.1007/978-3-031-73281-2_15 (paper)
#               https://doi.org/10.3929/ethz-c-000799752 (dataset)
# -----------------------------------------------------------------------------
import numpy as np
import pyvista as pv
import pyvista_zstd
import h5py
import glob
import re
import concurrent.futures
from tqdm import trange


# ---------------------------------------------------------------------------
# Quantisation helpers
# ---------------------------------------------------------------------------

def _quantize(arr: np.ndarray):
    """Min-max quantise a float array to uint16.

    Returns (q_uint16, scale, offset) such that arr ≈ q * scale + offset.
    """
    arr = arr.astype(np.float64)
    vmin = float(arr.min())
    vmax = float(arr.max())
    if vmax <= vmin:
        return np.zeros(arr.shape, dtype=np.uint16), 1.0, vmin
    scale = (vmax - vmin) / 65535.0
    q = np.round((arr - vmin) / scale).clip(0, 65535).astype(np.uint16)
    return q, scale, vmin


def _dequantize(q: np.ndarray, scale: float, offset: float) -> np.ndarray:
    """Inverse of _quantize. Returns float32 array."""
    return q.astype(np.float32) * np.float32(scale) + np.float32(offset)


# field_data keys written by compress_fields_pv that are bookkeeping for the
# uint16 quantisation, not part of the original file's parameters.
_FIELDS_BOOKKEEPING_RE = re.compile(r'_(scale|offset)_\d+$')


# ---------------------------------------------------------------------------
# Low-level single-file readers
# ---------------------------------------------------------------------------

def read_sensitivity(fpath):
    """Read a single sensitivity file and return the complex coil map.

    Returns
    -------
    np.ndarray, shape (*dimensions, n_coils), dtype complex
    """
    image = pyvista_zstd.read(str(fpath))
    dimensions = image.dimensions
    n_pts = np.prod(dimensions)
    real = image['real'].reshape((*dimensions, image['real'].size // n_pts), order='F')
    imag = image['imag'].reshape((*dimensions, image['imag'].size // n_pts), order='F')
    return real + 1j * imag


def read_displacement(fpath, key='breathing_state_to_end_expiration_ijk'):
    """Read a single displacement field file.

    Returns
    -------
    np.ndarray, shape (*dimensions, 3)
    """
    image = pyvista_zstd.read(str(fpath))
    dimensions = image.dimensions
    n_pts = np.prod(dimensions)
    n_comp = image[key].size // n_pts
    return image[key].reshape((*dimensions, n_comp), order='F')


def _read_single_frame(fpath):
    """Read one 4D-flow frame. Auto-detects float32, mag/phase uint16, and real/imag uint16.

    Returns
    -------
    dimensions : tuple of int
    spacing    : tuple of float
    mask       : np.ndarray, shape (*dimensions)
    signal     : np.ndarray, shape (*dimensions, n_comp), dtype complex
    parameters : dict
    """
    image = pyvista_zstd.read(str(fpath))
    dimensions = image.dimensions
    n_pts = int(np.prod(dimensions))

    mask = image['mask'].reshape(dimensions, order='F')

    try:
        parameters = dict(image.field_data)
    except Exception:
        parameters = {}

    if 'mag_uint16_0' in image.point_data:
        n_enc = sum(1 for k in image.point_data.keys() if k.startswith('mag_uint16_'))
        real = np.empty((*dimensions, n_enc), dtype=np.float32)
        imag = np.empty((*dimensions, n_enc), dtype=np.float32)
        for enc in range(n_enc):
            q_m = image.point_data[f'mag_uint16_{enc}'].reshape(dimensions, order='F')
            q_p = image.point_data[f'phase_uint16_{enc}'].reshape(dimensions, order='F')
            mags_r   = _dequantize(q_m,
                                   float(image.field_data[f'mag_scale_{enc}']),
                                   float(image.field_data[f'mag_offset_{enc}']))
            phases_r = _dequantize(q_p,
                                   float(image.field_data[f'phase_scale_{enc}']),
                                   float(image.field_data[f'phase_offset_{enc}']))
            real[..., enc] = mags_r * np.cos(phases_r)
            imag[..., enc] = mags_r * np.sin(phases_r)

    elif 'real_uint16_0' in image.point_data:
        n_enc = sum(1 for k in image.point_data.keys() if k.startswith('real_uint16_'))
        real = np.empty((*dimensions, n_enc), dtype=np.float32)
        imag = np.empty((*dimensions, n_enc), dtype=np.float32)
        for enc in range(n_enc):
            q_r = image.point_data[f'real_uint16_{enc}'].reshape(dimensions, order='F')
            q_i = image.point_data[f'imag_uint16_{enc}'].reshape(dimensions, order='F')
            real[..., enc] = _dequantize(q_r,
                                         float(image.field_data[f'real_scale_{enc}']),
                                         float(image.field_data[f'real_offset_{enc}']))
            imag[..., enc] = _dequantize(q_i,
                                         float(image.field_data[f'imag_scale_{enc}']),
                                         float(image.field_data[f'imag_offset_{enc}']))

    else:
        n_comp_real = image['real'].size // n_pts
        n_comp_imag = image['imag'].size // n_pts
        real = image['real'].reshape((*dimensions, n_comp_real), order='F')
        imag = image['imag'].reshape((*dimensions, n_comp_imag), order='F')

    return image.dimensions, image.spacing, mask, real + 1j * imag, parameters


def read_fields_pv(fpath):
    """Read a single .pv file's point-data fields. Transparent wrapper around
    _read_single_fields_pv.

    Works on any .pv file with arbitrary point_data fields (CFD ground-truth
    frames, sensitivity maps, displacement fields, …).

    Returns
    -------
    dimensions : tuple of int
    spacing    : tuple of float
    fields     : dict  field_name -> np.ndarray (float32)
                       shape (*dimensions) for scalars, (*dimensions, n_comp) for vectors
    parameters : dict  field_data (geometry matrices, acquisition parameters, …)
    """
    return _read_single_fields_pv(fpath)


def _read_single_fields_pv(fpath):
    """Read one .pv file's point-data fields. Auto-detects float32 and uint16
    compression (as written by compress_fields_pv).

    Returns
    -------
    dimensions : tuple of int
    spacing    : tuple of float
    fields     : dict  field_name -> np.ndarray (float32)
                       shape (*dimensions) for scalars, (*dimensions, n_comp) for vectors
    parameters : dict  field_data, excluding internal uint16 bookkeeping keys
    """
    image = pyvista_zstd.read(str(fpath))
    dimensions = image.dimensions
    n_pts = int(np.prod(dimensions))

    fields = {}

    is_compressed = ('fields_uint16' in image.field_data
                     and bool(image.field_data['fields_uint16'][0]))

    if is_compressed:
        compressed_keys = {k.rsplit('_uint16_', 1)[0]
                           for k in image.point_data.keys()
                           if '_uint16_' in k}
        for key in sorted(compressed_keys):
            n_comp = sum(1 for k in image.point_data.keys()
                         if k.startswith(f'{key}_uint16_'))
            if n_comp == 1:
                q = image.point_data[f'{key}_uint16_0'].reshape(dimensions, order='F')
                fields[key] = _dequantize(q,
                                          float(image.field_data[f'{key}_scale_0']),
                                          float(image.field_data[f'{key}_offset_0']))
            else:
                arr = np.empty((*dimensions, n_comp), dtype=np.float32)
                for comp in range(n_comp):
                    q = image.point_data[f'{key}_uint16_{comp}'].reshape(dimensions, order='F')
                    arr[..., comp] = _dequantize(q,
                                                 float(image.field_data[f'{key}_scale_{comp}']),
                                                 float(image.field_data[f'{key}_offset_{comp}']))
                fields[key] = arr
    else:
        for key in image.point_data.keys():
            arr = image.point_data[key].astype(np.float32)
            n_comp = arr.size // n_pts
            fields[key] = (arr.reshape((*dimensions, n_comp), order='F') if n_comp > 1
                           else arr.reshape(dimensions, order='F'))

    parameters = {k: v for k, v in image.field_data.items()
                   if k != 'fields_uint16' and not _FIELDS_BOOKKEEPING_RE.search(k)}

    return dimensions, image.spacing, fields, parameters


# ---------------------------------------------------------------------------
# Compression writers
# ---------------------------------------------------------------------------

def compress_mri_frame(fpath_in, fpath_out, mode='magphase'):
    """Read a float .pv MRI frame and write a uint16-compressed .pv file.

    Parameters
    ----------
    fpath_in  : path to the source float .pv file
    fpath_out : path for the output uint16 .pv file
    mode      : 'magphase' – quantises |signal| and angle(signal) independently (default);
                'realimag' – quantises real and imaginary parts independently.
    """
    image = pyvista_zstd.read(str(fpath_in))
    dimensions = image.dimensions
    n_pts = int(np.prod(dimensions))
    n_enc = image['real'].size // n_pts

    real = image['real'].reshape((*dimensions, n_enc), order='F').astype(np.float32)
    imag = image['imag'].reshape((*dimensions, n_enc), order='F').astype(np.float32)

    out = pv.ImageData()
    out.dimensions = dimensions
    out.spacing = image.spacing
    out.origin = image.origin

    if 'mask' in image.point_data:
        out.point_data['mask'] = image.point_data['mask']
    for k in image.field_data.keys():
        out.field_data[k] = image.field_data[k]

    if mode == 'magphase':
        sig    = real.astype(np.float64) + 1j * imag.astype(np.float64)
        mags   = np.abs(sig).astype(np.float32)
        phases = np.angle(sig).astype(np.float32)
        for enc in range(n_enc):
            q_m, s_m, o_m = _quantize(mags[..., enc])
            q_p, s_p, o_p = _quantize(phases[..., enc])
            out.point_data[f'mag_uint16_{enc}']   = q_m.ravel('F')
            out.point_data[f'phase_uint16_{enc}'] = q_p.ravel('F')
            out.field_data[f'mag_scale_{enc}']    = np.array([s_m])
            out.field_data[f'mag_offset_{enc}']   = np.array([o_m])
            out.field_data[f'phase_scale_{enc}']  = np.array([s_p])
            out.field_data[f'phase_offset_{enc}'] = np.array([o_p])

    elif mode == 'realimag':
        for enc in range(n_enc):
            q_r, s_r, o_r = _quantize(real[..., enc])
            q_i, s_i, o_i = _quantize(imag[..., enc])
            out.point_data[f'real_uint16_{enc}']  = q_r.ravel('F')
            out.point_data[f'imag_uint16_{enc}']  = q_i.ravel('F')
            out.field_data[f'real_scale_{enc}']   = np.array([s_r])
            out.field_data[f'real_offset_{enc}']  = np.array([o_r])
            out.field_data[f'imag_scale_{enc}']   = np.array([s_i])
            out.field_data[f'imag_offset_{enc}']  = np.array([o_i])

    else:
        raise ValueError(f"Unknown mode '{mode}'. Use 'magphase' or 'realimag'.")

    pyvista_zstd.write(out, str(fpath_out))


def compress_fields_pv(fpath_in, fpath_out):
    """Read a float .pv file and write a uint16-compressed .pv file.

    Works on any .pv file with arbitrary point_data fields (CFD ground truth,
    sensitivity maps, displacement fields, …); each field is quantised
    independently per component. field_data (geometry matrices, acquisition
    parameters, …) is preserved unchanged.
    """
    image = pyvista_zstd.read(str(fpath_in))
    dimensions = image.dimensions
    n_pts = int(np.prod(dimensions))

    out = pv.ImageData()
    out.dimensions = dimensions
    out.spacing = image.spacing
    out.origin = image.origin

    for k in image.field_data.keys():
        out.field_data[k] = image.field_data[k]
    out.field_data['fields_uint16'] = np.array([1], dtype=np.int32)

    for key in image.point_data.keys():
        arr = image.point_data[key].astype(np.float32)
        n_comp = arr.size // n_pts
        if n_comp > 1:
            arr_s = arr.reshape((*dimensions, n_comp), order='F')
            for comp in range(n_comp):
                q, s, o = _quantize(arr_s[..., comp])
                out.point_data[f'{key}_uint16_{comp}'] = q.ravel('F')
                out.field_data[f'{key}_scale_{comp}']  = np.array([s])
                out.field_data[f'{key}_offset_{comp}'] = np.array([o])
        else:
            arr_s = arr.reshape(dimensions, order='F')
            q, s, o = _quantize(arr_s)
            out.point_data[f'{key}_uint16_0'] = q.ravel('F')
            out.field_data[f'{key}_scale_0']  = np.array([s])
            out.field_data[f'{key}_offset_0'] = np.array([o])

    pyvista_zstd.write(out, str(fpath_out))


# ---------------------------------------------------------------------------
# Multi-frame readers
# ---------------------------------------------------------------------------

def read_4dflow_frames(pattern, max_workers=1):
    """Read all cardiac frames for one breathing bin into preallocated arrays.

    Parameters
    ----------
    pattern     : glob pattern selecting the .pv files for one bin
    max_workers : number of parallel reader threads

    Returns
    -------
    dimensions : tuple of int  (nx, ny, nz)
    spacing    : tuple of float
    mask       : np.ndarray, shape (*dimensions, n_cardiac_frames)
    signal     : np.ndarray, shape (*dimensions, n_comp, n_cardiac_frames), dtype complex
    parameters : dict  (taken from frame 0)
    """
    files = sorted(
        glob.glob(pattern),
        key=lambda f: int(re.search(r'_(\d+)\.pv$', f).group(1))
    )
    if not files:
        raise FileNotFoundError(f"No files matching '{pattern}' found.")

    n_frames = len(files)

    dimensions, spacing, mask0, signal0, parameters = _read_single_frame(files[0])

    mask   = np.empty((*dimensions, n_frames), dtype=mask0.dtype)
    signal = np.empty((*dimensions, signal0.shape[-1], n_frames), dtype=signal0.dtype)

    mask[..., 0]   = mask0
    signal[..., 0] = signal0

    if n_frames == 1:
        return dimensions, spacing, mask, signal, parameters

    def _load_frame(args):
        idx, fpath = args
        _, _, m, s, _ = _read_single_frame(fpath)
        mask[..., idx]   = m
        signal[..., idx] = s

    with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as executor:
        list(executor.map(_load_frame, enumerate(files[1:], start=1)))

    return dimensions, spacing, mask, signal, parameters


# ---------------------------------------------------------------------------
# PV writers
# ---------------------------------------------------------------------------

def save_4dflow_pv(out_dir, dims, spacing, mask, signal, parameters,
                   prefix='4DFlow', origin=(0.0, 0.0, 0.0)):
    """Write 4D-flow data to native .pv files (one per cardiac frame).

    Inverse of read_4dflow_frames: writes plain real/imag (float32) and the
    mask, uncompressed, plus all of `parameters` as field data — no uint16
    quantisation (see compress_mri_frame for that). Handy for saving a
    processed copy of the data in the same format it was loaded in.

    Parameters
    ----------
    out_dir    : output directory (created if it doesn't exist)
    dims       : tuple of int (nx, ny, nz), as returned by read_4dflow_frames
    spacing    : tuple of float
    mask       : np.ndarray, shape (*dims, n_frames)
    signal     : np.ndarray, shape (*dims, n_enc, n_frames), dtype complex
    parameters : dict, written verbatim as field data on every frame
    prefix     : output filename prefix -> '<prefix>_<NN>.pv'
    origin     : VTK ImageData origin. Not part of `parameters` (the RACLETTE
                 geometry lives in the 'ijk_to_xyz'/'xyz_to_raf' field data),
                 so it defaults to (0, 0, 0); pass the original image's
                 `.origin` to preserve it.

    Returns
    -------
    list of Path, the written .pv files (one per frame, sorted)
    """
    from pathlib import Path as _Path

    out_dir = _Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    n_pts    = int(np.prod(dims))
    n_enc    = signal.shape[3]
    n_frames = signal.shape[4]

    fpaths = []
    for f in range(n_frames):
        out = pv.ImageData()
        out.dimensions = dims
        out.spacing = spacing
        out.origin = origin

        out.point_data['mask'] = mask[..., f].ravel(order='F')
        out.point_data['real'] = (np.real(signal[..., f]).astype(np.float32)
                                   .reshape((n_pts, n_enc), order='F'))
        out.point_data['imag'] = (np.imag(signal[..., f]).astype(np.float32)
                                   .reshape((n_pts, n_enc), order='F'))

        for k, v in parameters.items():
            out.field_data[k] = np.asarray(v)

        fpath = out_dir / f'{prefix}_{f:02d}.pv'
        pyvista_zstd.write(out, str(fpath))
        fpaths.append(fpath)

    return fpaths


def save_fields_pv(fpath_out, dims, spacing, fields, parameters=None,
                   origin=(0.0, 0.0, 0.0)):
    """Write point-data fields (and optional field-data parameters) to a .pv file.

    Inverse of read_fields_pv: writes plain float32 point_data fields,
    uncompressed, plus `parameters` as field data — no uint16 quantisation
    (see compress_fields_pv for that). Works on the output of read_fields_pv /
    read_fields_pv_frames for any .pv file with arbitrary point_data fields
    (CFD ground truth, sensitivity maps, displacement fields, …).

    Parameters
    ----------
    fpath_out  : output .pv file path (parent directory created if needed)
    dims       : tuple of int (nx, ny, nz)
    spacing    : tuple of float
    fields     : dict  field_name -> np.ndarray
                       shape (*dims) for scalars, (*dims, n_comp) for vectors
    parameters : dict, optional, written verbatim as field data
    origin     : VTK ImageData origin

    Returns
    -------
    Path, the written .pv file
    """
    from pathlib import Path as _Path

    fpath_out = _Path(fpath_out)
    fpath_out.parent.mkdir(parents=True, exist_ok=True)

    n_pts = int(np.prod(dims))

    out = pv.ImageData()
    out.dimensions = dims
    out.spacing = spacing
    out.origin = origin

    for key, arr in fields.items():
        arr = np.asarray(arr, dtype=np.float32)
        if arr.ndim == len(dims):
            out.point_data[key] = arr.ravel(order='F')
        else:
            out.point_data[key] = arr.reshape((n_pts, arr.shape[-1]), order='F')

    if parameters:
        for k, v in parameters.items():
            out.field_data[k] = np.asarray(v)

    pyvista_zstd.write(out, str(fpath_out))

    return fpath_out


def read_fields_pv_frames(pattern, max_workers=4):
    """Read all .pv frames matching pattern into preallocated arrays.

    Parameters
    ----------
    pattern     : glob pattern selecting the .pv files
    max_workers : number of parallel reader threads

    Returns
    -------
    dimensions : tuple of int  (nx, ny, nz)
    spacing    : tuple of float
    fields     : dict  field_name -> np.ndarray
                       shape (*dimensions, n_frames) for scalars
                       shape (*dimensions, n_comp, n_frames) for vectors
    parameters : dict  field_data (taken from frame 0)
    """
    files = sorted(
        glob.glob(pattern),
        key=lambda f: int(re.search(r'_(\d+)\.pv$', f).group(1))
    )
    if not files:
        raise FileNotFoundError(f"No files matching '{pattern}' found.")

    n_frames = len(files)
    dimensions, spacing, fields0, parameters = _read_single_fields_pv(files[0])

    fields = {}
    for key, arr in fields0.items():
        if arr.ndim == len(dimensions):
            fields[key] = np.empty((*dimensions, n_frames), dtype=arr.dtype)
        else:
            fields[key] = np.empty((*dimensions, arr.shape[-1], n_frames), dtype=arr.dtype)
        fields[key][..., 0] = arr

    def _load_fields(args):
        idx, fpath = args
        _, _, f, _ = _read_single_fields_pv(fpath)
        for key, arr in f.items():
            fields[key][..., idx] = arr

    with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as executor:
        list(executor.map(_load_fields, enumerate(files[1:], start=1)))

    return dimensions, spacing, fields, parameters


# ---------------------------------------------------------------------------
# All-bins reader
# ---------------------------------------------------------------------------

def read_4dflow_all_bins(case_dir, n_breathing_bins, cardiac_pattern='7p_4DFlow_*.pv',
                         max_workers=6):
    """Read 4D-flow data for all breathing bins into a single preallocated array.

    Parameters
    ----------
    case_dir          : path to the case directory
    n_breathing_bins  : number of breathing bins
    cardiac_pattern   : glob pattern (filename only) for cardiac frames within a bin
    max_workers       : threads per bin

    Returns
    -------
    dimensions : tuple of int  (nx, ny, nz)
    spacing    : tuple of float
    mask       : np.ndarray, shape (*dimensions, n_cardiac_frames, n_breathing_bins)
    signal     : np.ndarray, shape (*dimensions, n_comp, n_cardiac_frames, n_breathing_bins)
    parameters : dict  (from bin 0, frame 0)
    """
    import os

    def _bin_pattern(i):
        return os.path.join(case_dir, f'Data/bin_{i:02d}', cardiac_pattern)

    dimensions, spacing, mask0, signal0, parameters = read_4dflow_frames(
        _bin_pattern(0), max_workers=max_workers
    )
    n_cardiac = mask0.shape[-1]
    n_comp    = signal0.shape[-2]

    mask   = np.empty((*dimensions, n_cardiac, n_breathing_bins), dtype=mask0.dtype)
    signal = np.empty((*dimensions, n_comp, n_cardiac, n_breathing_bins), dtype=signal0.dtype)

    mask[..., 0]   = mask0
    signal[..., 0] = signal0

    for i in trange(1, n_breathing_bins, leave=False, desc='Loading synthetic 4DFlow MR'):
        _, _, mask[..., i], signal[..., i], _ = read_4dflow_frames(
            _bin_pattern(i), max_workers=max_workers
        )

    return dimensions, spacing, mask, signal, parameters


# ---------------------------------------------------------------------------
# Displacement reader for all bins
# ---------------------------------------------------------------------------

def read_displacements_all_bins(case_dir, n_breathing_bins,
                                key='breathing_state_to_end_expiration_ijk'):
    """Read displacement fields for all breathing bins.

    Returns
    -------
    np.ndarray, shape (*dimensions, n_breathing_bins, 3)
    """
    import os

    disp0 = read_displacement(
        os.path.join(case_dir, 'Data/bin_00/breathing_motion.pv'), key=key
    )
    displacement = np.empty((*disp0.shape[:-1], n_breathing_bins, 3), dtype=disp0.dtype)
    displacement[..., 0, :] = disp0

    for i in trange(1, n_breathing_bins, leave=False, desc='Loading breathing motion'):
        fpath = os.path.join(case_dir, f'Data/bin_{i:02d}/breathing_motion.pv')
        displacement[..., i, :] = read_displacement(fpath, key=key)

    return displacement


# ---------------------------------------------------------------------------
# HDF5 I/O
# ---------------------------------------------------------------------------

def save_4dflow_h5(fpath, dims, spacing, mask, signal, parameters,
                   compression='gzip', compression_opts=4):
    """Save 4D-flow MRI data to an HDF5 file.

    Signal is stored as separate float32 real and imaginary datasets for
    maximum cross-tool compatibility (MATLAB, Julia, C++, …).

    Parameters
    ----------
    fpath            : output .h5 path
    dims             : (nx, ny, nz)
    spacing          : (dx, dy, dz)  [mm]
    mask             : (*dims, n_frames)        uint8/bool
    signal           : (*dims, n_enc, n_frames) complex
    parameters       : dict of acquisition metadata (kvs, vencs, …)
    compression      : HDF5 compression filter – 'gzip', 'lzf', or None
    compression_opts : gzip compression level 1–9 (ignored for lzf / None)
    """
    ckwargs = {'compression': compression}
    if compression == 'gzip' and compression_opts is not None:
        ckwargs['compression_opts'] = compression_opts

    with h5py.File(str(fpath), 'w') as f:
        f.attrs['description'] = '4D-flow MRI dataset – RACLETTE format'
        f.create_dataset('signal_real', data=signal.real.astype(np.float32), **ckwargs)
        f.create_dataset('signal_imag', data=signal.imag.astype(np.float32), **ckwargs)
        f.create_dataset('mask',        data=mask.astype(np.uint8),           **ckwargs)
        f.create_dataset('spacing',     data=np.array(spacing, dtype=np.float64))
        f.create_dataset('dims',        data=np.array(dims,    dtype=np.int32))
        grp = f.create_group('parameters')
        for key, val in parameters.items():
            try:
                grp.create_dataset(key, data=np.asarray(val))
            except Exception:
                pass


def load_4dflow_h5(fpath):
    """Load 4D-flow MRI data from an HDF5 file written by save_4dflow_h5.

    Returns
    -------
    dims       : tuple of int       (nx, ny, nz)
    spacing    : tuple of float     (dx, dy, dz)
    mask       : np.ndarray, shape (*dims, n_frames)        uint8
    signal     : np.ndarray, shape (*dims, n_enc, n_frames) complex64
    parameters : dict
    """
    with h5py.File(str(fpath), 'r') as f:
        signal = (f['signal_real'][:].astype(np.float32)
                  + 1j * f['signal_imag'][:].astype(np.float32))
        mask       = f['mask'][:]
        spacing    = tuple(float(v) for v in f['spacing'][:])
        dims       = tuple(int(v)   for v in f['dims'][:])
        parameters = {}
        if 'parameters' in f:
            for key in f['parameters']:
                parameters[key] = f['parameters'][key][:]
    return dims, spacing, mask, signal, parameters


def save_fields_h5(fpath, dims, spacing, fields, parameters=None,
                   compression='gzip', compression_opts=4):
    """Save point-data fields (and optional field-data parameters) to an HDF5 file.

    Works on the output of read_fields_pv / read_fields_pv_frames for any
    .pv file with arbitrary point_data fields (CFD ground truth, sensitivity
    maps, displacement fields, …).

    Parameters
    ----------
    fields     : dict  field_name → np.ndarray float32
                       shape (*dims) for scalars, (*dims, n_comp) for vectors
    parameters : dict, optional  field_data (geometry matrices, …), written
                 verbatim to a 'parameters' group
    """
    ckwargs = {'compression': compression}
    if compression == 'gzip' and compression_opts is not None:
        ckwargs['compression_opts'] = compression_opts

    with h5py.File(str(fpath), 'w') as f:
        f.attrs['description'] = 'Field dataset – RACLETTE format'
        f.create_dataset('spacing', data=np.array(spacing, dtype=np.float64))
        f.create_dataset('dims',    data=np.array(dims,    dtype=np.int32))
        grp = f.create_group('fields')
        for key, arr in fields.items():
            grp.create_dataset(key, data=arr.astype(np.float32), **ckwargs)
        if parameters:
            pgrp = f.create_group('parameters')
            for key, val in parameters.items():
                pgrp.create_dataset(key, data=np.asarray(val))


def load_fields_h5(fpath):
    """Load point-data fields (and field-data parameters) from an HDF5 file
    written by save_fields_h5.

    Returns
    -------
    dims       : tuple of int
    spacing    : tuple of float
    fields     : dict  field_name → np.ndarray float32
    parameters : dict
    """
    with h5py.File(str(fpath), 'r') as f:
        spacing = tuple(float(v) for v in f['spacing'][:])
        dims    = tuple(int(v)   for v in f['dims'][:])
        fields  = {key: f['fields'][key][:] for key in f['fields']}
        parameters = {}
        if 'parameters' in f:
            for key in f['parameters']:
                parameters[key] = f['parameters'][key][:]
    return dims, spacing, fields, parameters


# ---------------------------------------------------------------------------
# VTK/VTI I/O
# ---------------------------------------------------------------------------

def save_4dflow_vtk(out_dir, dims, spacing, mask, signal, parameters,
                     prefix='4DFlow', ext='.vtk', origin=(0.0, 0.0, 0.0),
                     **save_kwargs):
    """Write 4D-flow data to standard VTK files (one per cardiac frame).

    Same layout as save_4dflow_pv (mask + plain float32 real/imag +
    parameters as field data) but written with pyvista's standard VTK
    writers instead of the zstd-compressed .pv format, so the files are
    readable directly by ParaView and other VTK-based tools.

    Parameters
    ----------
    out_dir      : output directory (created if it doesn't exist)
    dims         : tuple of int (nx, ny, nz), as returned by read_4dflow_frames
    spacing      : tuple of float
    mask         : np.ndarray, shape (*dims, n_frames)
    signal       : np.ndarray, shape (*dims, n_enc, n_frames), dtype complex
    parameters   : dict, written verbatim as field data on every frame
    prefix       : output filename prefix -> '<prefix>_<NN><ext>'
    ext          : '.vtk' (legacy binary) or '.vti' (XML ImageData)
    origin       : VTK ImageData origin
    **save_kwargs: forwarded to pv.ImageData.save (e.g. compression='zlib'
                   for .vti; has no effect on .vtk)

    Returns
    -------
    list of Path, the written files (one per frame, sorted)
    """
    from pathlib import Path as _Path

    out_dir = _Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    n_pts    = int(np.prod(dims))
    n_enc    = signal.shape[3]
    n_frames = signal.shape[4]

    fpaths = []
    for f in range(n_frames):
        out = pv.ImageData()
        out.dimensions = dims
        out.spacing = spacing
        out.origin = origin

        out.point_data['mask'] = mask[..., f].ravel(order='F')
        out.point_data['real'] = (np.real(signal[..., f]).astype(np.float32)
                                   .reshape((n_pts, n_enc), order='F'))
        out.point_data['imag'] = (np.imag(signal[..., f]).astype(np.float32)
                                   .reshape((n_pts, n_enc), order='F'))

        for k, v in parameters.items():
            out.field_data[k] = np.asarray(v)

        fpath = out_dir / f'{prefix}_{f:02d}{ext}'
        out.save(str(fpath), **save_kwargs)
        fpaths.append(fpath)

    return fpaths


def _read_single_frame_vtk(fpath):
    """Read one 4D-flow frame from a plain .vtk/.vti file written by
    save_4dflow_vtk.

    Returns
    -------
    dimensions : tuple of int
    spacing    : tuple of float
    mask       : np.ndarray, shape (*dimensions)
    signal     : np.ndarray, shape (*dimensions, n_comp), dtype complex
    parameters : dict
    """
    image = pv.read(str(fpath))
    dimensions = image.dimensions
    n_pts = int(np.prod(dimensions))

    mask = image['mask'].reshape(dimensions, order='F')
    parameters = dict(image.field_data)

    n_comp_real = image['real'].size // n_pts
    n_comp_imag = image['imag'].size // n_pts
    real = image['real'].reshape((*dimensions, n_comp_real), order='F')
    imag = image['imag'].reshape((*dimensions, n_comp_imag), order='F')

    return dimensions, image.spacing, mask, real + 1j * imag, parameters


def read_4dflow_vtk_frames(pattern, max_workers=1):
    """Read all cardiac frames written by save_4dflow_vtk (.vtk or .vti).

    Same return signature as read_4dflow_frames, but reads plain VTK/VTI
    files via pyvista's standard readers instead of the zstd-compressed
    .pv format.

    Parameters
    ----------
    pattern     : glob pattern selecting the .vtk/.vti files for one bin
    max_workers : number of parallel reader threads

    Returns
    -------
    dimensions : tuple of int  (nx, ny, nz)
    spacing    : tuple of float
    mask       : np.ndarray, shape (*dimensions, n_cardiac_frames)
    signal     : np.ndarray, shape (*dimensions, n_comp, n_cardiac_frames), dtype complex
    parameters : dict  (taken from frame 0)
    """
    files = sorted(
        glob.glob(pattern),
        key=lambda f: int(re.search(r'_(\d+)\.(?:vtk|vti)$', f).group(1))
    )
    if not files:
        raise FileNotFoundError(f"No files matching '{pattern}' found.")

    n_frames = len(files)

    dimensions, spacing, mask0, signal0, parameters = _read_single_frame_vtk(files[0])

    mask   = np.empty((*dimensions, n_frames), dtype=mask0.dtype)
    signal = np.empty((*dimensions, signal0.shape[-1], n_frames), dtype=signal0.dtype)

    mask[..., 0]   = mask0
    signal[..., 0] = signal0

    if n_frames == 1:
        return dimensions, spacing, mask, signal, parameters

    def _load_frame(args):
        idx, fpath = args
        _, _, m, s, _ = _read_single_frame_vtk(fpath)
        mask[..., idx]   = m
        signal[..., idx] = s

    with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as executor:
        list(executor.map(_load_frame, enumerate(files[1:], start=1)))

    return dimensions, spacing, mask, signal, parameters


def save_fields_vtk(fpath_out, dims, spacing, fields, parameters=None,
                     origin=(0.0, 0.0, 0.0), **save_kwargs):
    """Write point-data fields (and optional field-data parameters) to a
    standard VTK file (.vtk or .vti, chosen by the extension of fpath_out).

    Same layout as save_fields_pv (plain float32 point_data fields plus
    parameters as field data) but written with pyvista's standard VTK
    writers, so the file is readable directly by ParaView and other
    VTK-based tools.

    Parameters
    ----------
    fpath_out    : output .vtk/.vti file path (parent directory created if needed)
    dims         : tuple of int (nx, ny, nz)
    spacing      : tuple of float
    fields       : dict  field_name -> np.ndarray
                         shape (*dims) for scalars, (*dims, n_comp) for vectors
    parameters   : dict, optional, written verbatim as field data
    origin       : VTK ImageData origin
    **save_kwargs: forwarded to pv.ImageData.save (e.g. compression='zlib'
                   for .vti; has no effect on .vtk)

    Returns
    -------
    Path, the written file
    """
    from pathlib import Path as _Path

    fpath_out = _Path(fpath_out)
    fpath_out.parent.mkdir(parents=True, exist_ok=True)

    n_pts = int(np.prod(dims))

    out = pv.ImageData()
    out.dimensions = dims
    out.spacing = spacing
    out.origin = origin

    for key, arr in fields.items():
        arr = np.asarray(arr, dtype=np.float32)
        if arr.ndim == len(dims):
            out.point_data[key] = arr.ravel(order='F')
        else:
            out.point_data[key] = arr.reshape((n_pts, arr.shape[-1]), order='F')

    if parameters:
        for k, v in parameters.items():
            out.field_data[k] = np.asarray(v)

    out.save(str(fpath_out), **save_kwargs)

    return fpath_out


def read_fields_vtk(fpath):
    """Read a single .vtk/.vti file's point-data fields, written by
    save_fields_vtk.

    Returns
    -------
    dimensions : tuple of int
    spacing    : tuple of float
    fields     : dict  field_name -> np.ndarray (float32)
                       shape (*dimensions) for scalars, (*dimensions, n_comp) for vectors
    parameters : dict  field_data (geometry matrices, acquisition parameters, …)
    """
    image = pv.read(str(fpath))
    dimensions = image.dimensions
    n_pts = int(np.prod(dimensions))

    fields = {}
    for key in image.point_data.keys():
        arr = image.point_data[key].astype(np.float32)
        n_comp = arr.size // n_pts
        fields[key] = (arr.reshape((*dimensions, n_comp), order='F') if n_comp > 1
                       else arr.reshape(dimensions, order='F'))

    parameters = dict(image.field_data)

    return dimensions, image.spacing, fields, parameters


# ---------------------------------------------------------------------------
# Geometry helpers
# ---------------------------------------------------------------------------

def _corrected_ijk_to_xyz(spacing, parameters):
    """Build the (4,4) IJK -> scanner-XYZ affine in metres, with correct
    per-axis voxel spacing.

    By design, ijk_to_xyz / xyz_to_ijk only encode *rotation* (their linear
    part has unit-norm orthogonal columns once normalised) plus the volume
    *origin* (translation column, in m); going from a voxel index to a
    physical position additionally requires scaling by the per-axis voxel
    spacing, which RACLETTE stores separately. The linear part is therefore
    rebuilt as R @ diag(spacing), where R is ijk_to_xyz's rotation with
    unit-length columns.
    """
    ijk_to_xyz = np.asarray(parameters['ijk_to_xyz'], dtype=np.float64)
    R3 = ijk_to_xyz[:3, :3]
    R  = R3 / np.linalg.norm(R3, axis=0)
    out = np.eye(4)
    out[:3, :3] = R @ np.diag(spacing)
    out[:3, 3]  = ijk_to_xyz[:3, 3]
    return out


def _ijk_to_lps(spacing, parameters):
    """Build the (4,4) IJK -> patient-LPS affine in mm.

    ijk_to_xyz only encodes rotation + origin (see _corrected_ijk_to_xyz);
    the per-axis voxel spacing is reintroduced before converting to LPS.

    Falls back to a diagonal spacing affine when ijk_to_xyz / xyz_to_raf are
    absent from parameters (mirrors save_4dflow_nifti's fallback).
    """
    if 'ijk_to_xyz' in parameters and 'xyz_to_raf' in parameters:
        ijk_to_xyz = _corrected_ijk_to_xyz(spacing, parameters)
        xyz_to_raf = np.asarray(parameters['xyz_to_raf'], dtype=np.float64)

        ijk_to_lps = xyz_to_raf @ ijk_to_xyz
        ijk_to_lps[:3, :] *= 1000.0          # m -> mm
    else:
        ijk_to_lps = np.diag([*(np.array(spacing) * 1000.0), 1.0])
    return ijk_to_lps


# ---------------------------------------------------------------------------
# NIfTI I/O
# ---------------------------------------------------------------------------
# nibabel is imported lazily so it is only required when NIfTI I/O is used.
#
# File layout written by save_4dflow_nifti:
#
#   <out_dir>/
#     mask.nii[.gz]              (*dims, n_frames)   uint8
#     magnitude_enc0.nii[.gz]   (*dims, n_frames)   float32   ← all n_enc
#     magnitude_enc1.nii[.gz]
#     …
#     phase_enc0.nii[.gz]       (*dims, n_frames)   float32   ← all n_enc
#     phase_enc1.nii[.gz]
#     …
#     velocity_dir0.nii[.gz]    (*dims, n_frames)   float32   ← n_enc-1 dirs
#     velocity_dir1.nii[.gz]                                    (if kvs present)
#     …
#     parameters.json            acquisition metadata (kvs, vencs, …)
#
# Note: load_4dflow_nifti reconstructs complex signal as mag*exp(i*phase),
# which introduces small float32 rounding errors vs the original real+i*imag.
# ---------------------------------------------------------------------------

def save_4dflow_nifti(out_dir, dims, spacing, mask, signal, parameters,
                      compressed=True):
    """Save 4D-flow MRI data as NIfTI files (one per encoding / direction).

    Coordinate convention
    ---------------------
    The RACLETTE / GyroTools xyz_to_raf matrix follows the GyroTools convention
    (https://docs.gyrotools.com/mrecon/latest/src/geo/geometry_transformations.html):

        RAF (GyroTools)  →  positive axis direction
          R  :  Right  → Left          (RL)
          A  :  Anterior → Posterior   (AP)
          F  :  Feet  → Head           (FH)

    The NIfTI standard and its ecosystem (nibabel, ITK-SNAP, FSL) expect world
    coordinates in RAS convention:

        RAS (NIfTI)  →  positive axis direction
          R  :  Left  → Right          (LR)   ← opposite to RAF R
          A  :  Posterior → Anterior   (PA)   ← opposite to RAF A
          S  :  Feet  → Head           (FH/IS)  ← same as RAF F

    The conversion is therefore a sign-flip of the first two axes:

        xyz_to_ras = diag([-1, -1, 1, 1]) @ xyz_to_raf

    Without this correction NIfTI tools misinterpret the R and A axes and
    report the wrong orientation (SAR instead of SPL for this dataset).

    The NIfTI affine is then:

        voxel ──[ijk_to_xyz]──► scanner XYZ ──[xyz_to_ras]──► RAS (mm)

    enc_dirs are rotated to RAS so they are consistent with the NIfTI world
    space.  kvs and vencs are scalar sensitivities and are saved unchanged.

    Falls back to a simple diagonal spacing affine when ijk_to_xyz or
    xyz_to_raf are absent from parameters.

    Parameters
    ----------
    out_dir    : directory that will receive the NIfTI files (created if absent)
    dims       : (nx, ny, nz)
    spacing    : (dx, dy, dz)  [m]  – used only as fallback when matrices absent
    mask       : (*dims, n_frames)        uint8/bool
    signal     : (*dims, n_enc, n_frames) complex
    parameters : dict of acquisition metadata (kvs, vencs, enc_dirs,
                 ijk_to_xyz, xyz_to_raf, xyz_to_ijk, …)
    compressed : True → .nii.gz  (gzip),  False → .nii  (uncompressed)
    """
    import json
    import nibabel as nib
    from pathlib import Path as _Path

    out_dir = _Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    ext   = '.nii.gz' if compressed else '.nii'
    n_enc = signal.shape[3]

    # ── Build IJK → RAS affine (mm) ──────────────────────────────────────────
    # xyz_to_raf (GyroTools convention): R+ = Left (RL), A+ = Posterior (AP),
    #                                    F+ = Head (FH).
    # NIfTI / RAS convention:            R+ = Right (LR), A+ = Anterior (PA),
    #                                    S+ = Head (FH/IS).
    # R and A axes are sign-inverted between the two conventions; F/S is the
    # same.  Negating the first two rows of xyz_to_raf gives xyz_to_ras:
    #   xyz_to_ras = diag([-1, -1, 1, 1]) @ xyz_to_raf
    # Without this correction NIfTI tools (ITK-SNAP, FSL, nibabel) misinterpret
    # the R and A axes and report the wrong orientation (SAR instead of SPL).
    #
    # ijk_to_xyz only encodes rotation + origin (see _corrected_ijk_to_xyz);
    # the per-axis voxel spacing is reintroduced before building the affine.
    if 'ijk_to_xyz' in parameters and 'xyz_to_raf' in parameters:
        ijk_to_xyz = _corrected_ijk_to_xyz(spacing, parameters)
        xyz_to_raf = np.asarray(parameters['xyz_to_raf'], dtype=np.float64)
        xyz_to_ras = np.diag([-1., -1., 1., 1.]) @ xyz_to_raf
        affine     = xyz_to_ras @ ijk_to_xyz
        affine[:3, :] *= 1000.0                             # m → mm
        R_to_ras   = xyz_to_ras[:3, :3]
    else:
        affine   = np.diag([*(np.array(spacing) * 1000.0), 1.0])
        R_to_ras = np.eye(3)

    # ── Serialise parameters; rotate enc_dirs to RAS ──────────────────────────
    # enc_dirs are direction vectors in scanner XYZ – rotate them so they are
    # expressed in the same RAS world space as the NIfTI affine.
    params_out = {}
    for key, val in parameters.items():
        try:
            params_out[key] = np.asarray(val).tolist()
        except Exception:
            pass
    if 'enc_dirs' in parameters:
        enc_dirs_ras = (R_to_ras @ np.asarray(parameters['enc_dirs']).T).T
        params_out['enc_dirs'] = enc_dirs_ras.tolist()

    # ── Helper: create and save one NIfTI image ──────────────────────────────
    def _save_nii(data, fname):
        img = nib.Nifti1Image(data, affine)
        img.header.set_xyzt_units('mm')
        nib.save(img, str(out_dir / fname))

    # ── Mask ─────────────────────────────────────────────────────────────────
    _save_nii(mask.astype(np.uint8), f'mask{ext}')

    # ── Magnitude and phase – one file per encoding direction ─────────────────
    for i in range(n_enc):
        sig_i = signal[:, :, :, i, :]
        _save_nii(np.abs(sig_i).astype(np.float32),   f'magnitude_enc{i}{ext}')
        _save_nii(np.angle(sig_i).astype(np.float32), f'phase_enc{i}{ext}')

    # ── Velocity – one file per flow-encoded direction ────────────────────────
    if 'kvs' in parameters:
        from flowProcessing import extract_velocities
        kvs = np.asarray(parameters['kvs']).ravel().astype(np.float64)
        vel = extract_velocities(signal, kvs)           # (*dims, n_enc-1, n_frames)
        for i in range(vel.shape[3]):
            _save_nii(vel[:, :, :, i, :].astype(np.float32), f'velocity_dir{i}{ext}')

    # ── Parameters JSON sidecar ───────────────────────────────────────────────
    with open(out_dir / 'parameters.json', 'w') as fj:
        json.dump(params_out, fj, indent=2)


def load_4dflow_nifti(out_dir, compressed=True):
    """Load 4D-flow MRI data from NIfTI files written by save_4dflow_nifti.

    Reconstructs complex signal as magnitude * exp(i * phase).

    Returns
    -------
    dims       : tuple of int       (nx, ny, nz)
    spacing    : tuple of float     (dx, dy, dz)
    mask       : np.ndarray, shape (*dims, n_frames)        uint8
    signal     : np.ndarray, shape (*dims, n_enc, n_frames) complex64
    parameters : dict
    """
    import json
    import nibabel as nib
    from pathlib import Path as _Path

    out_dir = _Path(out_dir)
    ext     = '.nii.gz' if compressed else '.nii'

    mag_files = sorted(
        out_dir.glob(f'magnitude_enc*{ext}'),
        key=lambda p: int(re.search(r'enc(\d+)', p.name).group(1)),
    )
    if not mag_files:
        raise FileNotFoundError(f"No magnitude_enc* files found in {out_dir}")

    n_enc  = len(mag_files)
    first  = nib.load(str(mag_files[0]))
    affine = first.affine
    sh     = first.shape                          # (*dims, n_frames)
    dims   = sh[:3]
    n_frames = sh[3] if len(sh) > 3 else 1

    signal = np.empty((*dims, n_enc, n_frames), dtype=np.complex64)
    for i, mf in enumerate(mag_files):
        mag   = nib.load(str(mf)).get_fdata(dtype=np.float32)
        phase = nib.load(str(out_dir / f'phase_enc{i}{ext}')).get_fdata(dtype=np.float32)
        signal[..., i, :] = mag * (np.cos(phase) + 1j * np.sin(phase))

    mask    = nib.load(str(out_dir / f'mask{ext}')).get_fdata(dtype=np.float32).astype(np.uint8)
    # Spacing from affine column norms (handles non-diagonal / rotated affines)
    # Convert mm back to m to stay consistent with the rest of the pipeline.
    spacing = tuple(float(np.linalg.norm(affine[:3, i])) / 1000.0 for i in range(3))

    parameters = {}
    params_path = out_dir / 'parameters.json'
    if params_path.exists():
        with open(params_path) as fj:
            parameters = {k: np.array(v) for k, v in json.load(fj).items()}

    return dims, spacing, mask, signal, parameters


# ---------------------------------------------------------------------------
# NRRD I/O
# ---------------------------------------------------------------------------
# pynrrd is imported lazily so it is only required when NRRD I/O is used.
#
# File layout written by save_4dflow_nrrd (one .nrrd per quantity, mirroring
# save_4dflow_nifti's per-quantity layout):
#
#   <out_dir>/
#     mask.nrrd                  (*dims, n_frames)   uint8
#     magnitude_enc0.nrrd       (*dims, n_frames)   float32   ← all n_enc
#     magnitude_enc1.nrrd
#     …
#     phase_enc0.nrrd           (*dims, n_frames)   float32   ← all n_enc
#     phase_enc1.nrrd
#     …
#     velocity_dir0.nrrd        (*dims, n_frames)   float32   ← n_enc-1 dirs
#     velocity_dir1.nrrd                                          (if kvs present)
#     …
#     parameters.json             acquisition metadata (kvs, vencs, enc_dirs
#                                  rotated to LPS, …)
#
# Geometry note: NRRD's "left-posterior-superior" space is identical to
# xyz_to_raf (GyroTools RAF ≡ DICOM LPS, see the DICOM I/O section below) —
# like the DICOM export, and unlike the NIfTI export's RAS, no axis
# sign-flip is required:
#
#   ijk_to_lps = xyz_to_raf @ ijk_to_xyz   (m → mm)
#
# This (4,4) affine maps directly onto NRRD's "space directions" (the three
# spatial-axis columns, with per-axis voxel spacing already baked in by
# _corrected_ijk_to_xyz) and "space origin" (its translation column).
# enc_dirs are rotated by xyz_to_raf's rotation part to remain consistent
# with this LPS world space.
#
# Each .nrrd is a single 4D array: 3 spatial "domain" axes plus one "list"
# axis for the cardiac frames (the convention used by 3D Slicer's
# Sequences/MultiVolume readers for 4D NRRD). Data is stored losslessly as
# float32 / uint8 — no quantisation, in contrast to the DICOM export's
# uint16 + RescaleSlope/Intercept.
#
# encoding ('raw' vs 'gzip') is an internal NRRD header field; both are read
# back transparently by nrrd.read, unlike NIfTI's .nii / .nii.gz extension
# difference.
#
# Note: load_4dflow_nrrd reconstructs complex signal as mag*exp(i*phase),
# which introduces the same small float32 rounding error noted for
# load_4dflow_nifti.
# ---------------------------------------------------------------------------

def save_4dflow_nrrd(out_dir, dims, spacing, mask, signal, parameters,
                     compressed=True):
    """Save 4D-flow MRI data as NRRD files (one per encoding / direction),
    oriented to patient LPS coordinates.

    See the module-level "NRRD I/O" comment block for the file layout and
    the LPS/RAF geometry note (no axis sign-flip vs scanner XYZ, unlike the
    NIfTI export's RAS).

    Parameters
    ----------
    out_dir    : directory that will receive the NRRD files (created if absent)
    dims       : (nx, ny, nz)
    spacing    : (dx, dy, dz)  [m]  – used only as fallback when ijk_to_xyz /
                 xyz_to_raf are absent from parameters
    mask       : (*dims, n_frames)        uint8/bool
    signal     : (*dims, n_enc, n_frames) complex
    parameters : dict of acquisition metadata (kvs, vencs, enc_dirs,
                 ijk_to_xyz, xyz_to_raf, …)
    compressed : True -> 'gzip' encoding, False -> 'raw' encoding (both
                 stored in the same .nrrd extension; transparent on read)
    """
    import json
    import nrrd
    from pathlib import Path as _Path

    out_dir = _Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    n_enc = signal.shape[3]

    # ── Build IJK -> LPS affine (mm) and NRRD orientation header fields ──────
    ijk_to_lps = _ijk_to_lps(spacing, parameters)
    space_directions = list(ijk_to_lps[:3, :3].T) + [np.full(3, np.nan)]
    space_origin     = ijk_to_lps[:3, 3]

    base_header = {
        'space': 'left-posterior-superior',
        'space directions': space_directions,
        'space origin': space_origin,
        'space units': ['mm', 'mm', 'mm'],
        'kinds': ['domain', 'domain', 'domain', 'list'],
        'encoding': 'gzip' if compressed else 'raw',
    }

    # ── Serialise parameters; rotate enc_dirs to LPS ─────────────────────────
    # enc_dirs are direction vectors in scanner XYZ – rotate them so they are
    # expressed in the same LPS world space as "space directions"/"space origin".
    params_out = {}
    for key, val in parameters.items():
        try:
            params_out[key] = np.asarray(val).tolist()
        except Exception:
            pass
    if 'enc_dirs' in parameters and 'xyz_to_raf' in parameters:
        R_to_lps = np.asarray(parameters['xyz_to_raf'], dtype=np.float64)[:3, :3]
        enc_dirs_lps = (R_to_lps @ np.asarray(parameters['enc_dirs']).T).T
        params_out['enc_dirs'] = enc_dirs_lps.tolist()

    # ── Helper: write one 4D NRRD volume ─────────────────────────────────────
    def _save_nrrd(data, fname):
        nrrd.write(str(out_dir / fname), data, header=dict(base_header))

    # ── Mask ──────────────────────────────────────────────────────────────────
    _save_nrrd(mask.astype(np.uint8), 'mask.nrrd')

    # ── Magnitude and phase – one file per encoding direction ─────────────────
    for i in range(n_enc):
        sig_i = signal[:, :, :, i, :]
        _save_nrrd(np.abs(sig_i).astype(np.float32),   f'magnitude_enc{i}.nrrd')
        _save_nrrd(np.angle(sig_i).astype(np.float32), f'phase_enc{i}.nrrd')

    # ── Velocity – one file per flow-encoded direction ─────────────────────────
    if 'kvs' in parameters:
        from flowProcessing import extract_velocities
        kvs = np.asarray(parameters['kvs']).ravel().astype(np.float64)
        vel = extract_velocities(signal, kvs)           # (*dims, n_enc-1, n_frames)
        for i in range(vel.shape[3]):
            _save_nrrd(vel[:, :, :, i, :].astype(np.float32), f'velocity_dir{i}.nrrd')

    # ── Parameters JSON sidecar ───────────────────────────────────────────────
    with open(out_dir / 'parameters.json', 'w') as fj:
        json.dump(params_out, fj, indent=2)


def load_4dflow_nrrd(out_dir):
    """Load 4D-flow MRI data from NRRD files written by save_4dflow_nrrd.

    Reconstructs complex signal as magnitude * exp(i * phase).

    Returns
    -------
    dims       : tuple of int       (nx, ny, nz)
    spacing    : tuple of float     (dx, dy, dz)
    mask       : np.ndarray, shape (*dims, n_frames)        uint8
    signal     : np.ndarray, shape (*dims, n_enc, n_frames) complex64
    parameters : dict
    """
    import json
    import nrrd
    from pathlib import Path as _Path

    out_dir = _Path(out_dir)

    mag_files = sorted(
        out_dir.glob('magnitude_enc*.nrrd'),
        key=lambda p: int(re.search(r'enc(\d+)', p.name).group(1)),
    )
    if not mag_files:
        raise FileNotFoundError(f"No magnitude_enc* files found in {out_dir}")

    n_enc = len(mag_files)
    first, header = nrrd.read(str(mag_files[0]))
    dims     = first.shape[:3]
    n_frames = first.shape[3] if first.ndim > 3 else 1

    # Spacing from "space directions" column norms (handles rotated affines).
    space_dirs = np.asarray(header['space directions'][:3], dtype=np.float64)
    spacing = tuple(float(np.linalg.norm(space_dirs[i])) / 1000.0 for i in range(3))  # mm -> m

    signal = np.empty((*dims, n_enc, n_frames), dtype=np.complex64)
    for i, mf in enumerate(mag_files):
        mag,   _ = nrrd.read(str(mf))
        phase, _ = nrrd.read(str(out_dir / f'phase_enc{i}.nrrd'))
        signal[..., i, :] = mag * (np.cos(phase) + 1j * np.sin(phase))

    mask, _ = nrrd.read(str(out_dir / 'mask.nrrd'))

    parameters = {}
    params_path = out_dir / 'parameters.json'
    if params_path.exists():
        with open(params_path) as fj:
            parameters = {k: np.array(v) for k, v in json.load(fj).items()}

    return dims, spacing, mask, signal, parameters


# ---------------------------------------------------------------------------
# DICOM I/O
# ---------------------------------------------------------------------------
# pydicom is imported lazily so it is only required when DICOM I/O is used.
#
# File layout written by save_4dflow_dicom:
#
#   <out_dir>/
#     mask/
#       phase000.dcm .. phase{N-1}.dcm   (nx,ny,nz) uint16 + rescale (exact 0/1)
#     magnitude_enc0/
#       phase000.dcm .. phase{N-1}.dcm   ← all n_enc
#     magnitude_enc1/
#     …
#     phase_enc0/
#       phase000.dcm .. phase{N-1}.dcm   ← all n_enc
#     phase_enc1/
#     …
#     velocity_dir0/
#       phase000.dcm .. phase{N-1}.dcm   ← n_enc-1 dirs (if kvs present)
#     velocity_dir1/
#     …
#     parameters.json            acquisition metadata, dims, spacing,
#                                 per-quantity {scale, offset}
#
# Each phaseNNN.dcm is a single-stack ('3D') Enhanced MR Image Storage
# (multi-frame) dataset with NumberOfFrames = nz – one ordinary 3D volume per
# cardiac phase. Every (quantity, phase) pair gets its own
# SeriesInstanceUID/SeriesNumber, while StudyInstanceUID/FrameOfReferenceUID
# are shared across the whole export: a "set of 3D images, one series per
# time point, sharing a study" – the convention recommended for 4D DICOM at
# https://discourse.itk.org/t/ordering-of-4d-dicom-images-while-writing/5277
# (see also https://github.com/SimpleITK/SimpleITK/issues/879). Each phase's
# temporal position is identified redundantly via TemporalPositionIdentifier /
# NumberOfTemporalPositions, ContentTime and TriggerTime (spaced by TR from
# ms_parameters) so that both "series-grouping" and "instance-tag" 4D readers
# can reassemble the cine.
#
# Pixel data is stored as uint16 with a RescaleSlope/RescaleIntercept obtained
# from _quantize over the whole (*dims, n_frames) quantity (shared across all
# phases for consistent windowing) – the same quantisation explored in the
# Uint16Compression tutorial. The mask roundtrips exactly (0/1 → 0/65535 →
# 0.0/1.0); magnitude/phase/velocity acquire the same small quantisation error.
#
# Geometry note: xyz_to_raf already matches DICOM's patient LPS convention
# (R+ = Left, A+ = Posterior, F+ = Head ≡ DICOM L, P, S). Unlike the NIfTI
# export (RAS), no axis sign-flip is required: ijk_to_lps = xyz_to_raf @ ijk_to_xyz.
#
# Note: load_4dflow_dicom reconstructs complex signal as mag*exp(i*phase),
# which – on top of the uint16 quantisation – introduces the same small
# float32 rounding error noted for load_4dflow_nifti.
# ---------------------------------------------------------------------------

def _dicom_geometry(ijk_to_lps, n_slices):
    """Derive PixelSpacing, SliceThickness, ImageOrientationPatient and the
    per-slice ImagePositionPatient list from an IJK->LPS affine (mm)."""
    col_i, col_j, col_k = ijk_to_lps[:3, 0], ijk_to_lps[:3, 1], ijk_to_lps[:3, 2]
    spacing_i = float(np.linalg.norm(col_i))
    spacing_j = float(np.linalg.norm(col_j))
    spacing_k = float(np.linalg.norm(col_k))

    orientation = [*(col_i / spacing_i), *(col_j / spacing_j)]
    pixel_spacing = [spacing_j, spacing_i]   # DICOM order: (row spacing, column spacing)

    positions = [(ijk_to_lps @ np.array([0., 0., k, 1.]))[:3] for k in range(n_slices)]
    return pixel_spacing, spacing_k, orientation, positions


def _ms_parameters_dict(parameters):
    """Parse the flat ['key', 'value', 'key', 'value', ...] ms_parameters
    array (as stored in parameters.json) into a {key: float(value)} dict."""
    raw = parameters.get('ms_parameters')
    if raw is None:
        return {}
    flat = np.asarray(raw).ravel().tolist()
    return {flat[i]: float(flat[i + 1]) for i in range(0, len(flat) - 1, 2)}


def _write_dicom_volume(fpath, q_volume, *, pixel_spacing, slice_thickness,
                        orientation, positions, scale, offset,
                        series_uid, series_number, series_description,
                        study_uid, frame_of_reference_uid,
                        complex_image_component, transfer_syntax,
                        temporal_position_index, n_temporal_positions,
                        content_time, trigger_time):
    """Write one cardiac phase of one quantity as a single-stack ('3D')
    multi-frame Enhanced MR Image: NumberOfFrames = nz, one frame per slice –
    an ordinary 3D volume.

    q_volume must already be quantised to uint16 (shape (nx, ny, nz)); scale
    /offset are the RescaleSlope/RescaleIntercept for the *whole* quantity
    (shared across phases so windowing is consistent across the cardiac cine).
    """
    import datetime
    from pydicom.dataset import FileDataset, FileMetaDataset, Dataset
    from pydicom.sequence import Sequence
    from pydicom.tag import Tag
    from pydicom.uid import EnhancedMRImageStorage, generate_uid

    nx, ny, nz = q_volume.shape
    rows, cols = ny, nx

    sop_instance_uid = generate_uid()
    file_meta = FileMetaDataset()
    file_meta.MediaStorageSOPClassUID = EnhancedMRImageStorage
    file_meta.MediaStorageSOPInstanceUID = sop_instance_uid
    file_meta.TransferSyntaxUID = transfer_syntax

    ds = FileDataset(str(fpath), {}, file_meta=file_meta, preamble=b'\x00' * 128)
    ds.is_little_endian = True
    ds.is_implicit_VR = False

    now = datetime.datetime.now()
    date_str = now.strftime('%Y%m%d')

    # SOP Common
    ds.SOPClassUID = EnhancedMRImageStorage
    ds.SOPInstanceUID = sop_instance_uid
    ds.ContentDate = date_str
    ds.ContentTime = content_time
    ds.InstanceNumber = 1

    # Patient / General Study (placeholder values – example data only)
    ds.PatientName = 'RACLETTE^Example'
    ds.PatientID = 'RACLETTE001'
    ds.PatientBirthDate = ''
    ds.PatientSex = ''
    ds.StudyInstanceUID = study_uid
    ds.StudyDate = date_str
    ds.StudyTime = now.strftime('%H%M%S')
    ds.StudyID = '1'
    ds.AccessionNumber = ''
    ds.ReferringPhysicianName = ''

    # General Series – one series per (quantity, cardiac phase) "3D image"
    ds.SeriesInstanceUID = series_uid
    ds.SeriesNumber = series_number
    ds.Modality = 'MR'
    ds.SeriesDescription = series_description

    # Cardiac/temporal position of this 3D volume within the cine
    ds.TemporalPositionIdentifier = temporal_position_index
    ds.NumberOfTemporalPositions = n_temporal_positions
    ds.TriggerTime = trigger_time

    # Frame of Reference
    ds.FrameOfReferenceUID = frame_of_reference_uid
    ds.PositionReferenceIndicator = ''

    # Enhanced General Equipment (mandatory for Enhanced MR Image)
    ds.Manufacturer = 'RACLETTE'
    ds.ManufacturerModelName = 'RACLETTE_examples'
    ds.DeviceSerialNumber = '0'
    ds.SoftwareVersions = '1.0'

    # Enhanced MR Image (top level)
    ds.ImageType = ['ORIGINAL', 'PRIMARY', 'OTHER']
    ds.AcquisitionContextSequence = Sequence([])

    # Image Pixel
    ds.SamplesPerPixel = 1
    ds.PhotometricInterpretation = 'MONOCHROME2'
    ds.Rows = rows
    ds.Columns = cols
    ds.BitsAllocated = 16
    ds.BitsStored = 16
    ds.HighBit = 15
    ds.PixelRepresentation = 0
    ds.NumberOfFrames = nz

    # ── Shared Functional Groups (same for every frame) ──────────────────────
    shared = Dataset()

    pixel_measures = Dataset()
    pixel_measures.PixelSpacing = [float(v) for v in pixel_spacing]
    pixel_measures.SliceThickness = float(slice_thickness)
    shared.PixelMeasuresSequence = Sequence([pixel_measures])

    plane_orientation = Dataset()
    plane_orientation.ImageOrientationPatient = [float(v) for v in orientation]
    shared.PlaneOrientationSequence = Sequence([plane_orientation])

    pixel_value_transform = Dataset()
    pixel_value_transform.RescaleIntercept = float(offset)
    pixel_value_transform.RescaleSlope = float(scale)
    pixel_value_transform.RescaleType = 'US'
    shared.PixelValueTransformationSequence = Sequence([pixel_value_transform])

    mr_frame_type = Dataset()
    mr_frame_type.FrameType = ['ORIGINAL', 'PRIMARY', 'OTHER', 'NONE']
    mr_frame_type.PixelPresentation = 'MONOCHROME'
    mr_frame_type.VolumetricProperties = 'VOLUME'
    mr_frame_type.VolumeBasedCalculationTechnique = 'NONE'
    mr_frame_type.ComplexImageComponent = complex_image_component
    mr_frame_type.AcquisitionContrast = 'UNKNOWN'
    shared.MRImageFrameTypeSequence = Sequence([mr_frame_type])

    ds.SharedFunctionalGroupsSequence = Sequence([shared])

    # ── Per-Frame Functional Groups: single 3D stack, one frame per slice ───
    pixel_frames = np.empty((nz, rows, cols), dtype=np.uint16)
    per_frame_items = []
    for s in range(nz):
        item = Dataset()

        plane_position = Dataset()
        plane_position.ImagePositionPatient = [float(v) for v in positions[s]]
        item.PlanePositionSequence = Sequence([plane_position])

        frame_content = Dataset()
        frame_content.DimensionIndexValues = [s + 1]
        frame_content.StackID = '1'
        frame_content.InStackPositionNumber = s + 1
        item.FrameContentSequence = Sequence([frame_content])

        per_frame_items.append(item)
        pixel_frames[s] = q_volume[:, :, s].T

    ds.PerFrameFunctionalGroupsSequence = Sequence(per_frame_items)

    # ── Multi-frame Dimension Module ──────────────────────────────────────────
    # DimensionOrganizationType='3D': a single ordered stack of frames indexed
    # by InStackPositionNumber – the simplest, most broadly supported Enhanced
    # MR organisation (one ordinary 3D volume per file/series).
    ds.DimensionOrganizationType = '3D'

    dimension_organization_uid = generate_uid()
    dim_org = Dataset()
    dim_org.DimensionOrganizationUID = dimension_organization_uid
    ds.DimensionOrganizationSequence = Sequence([dim_org])

    di_stack = Dataset()
    di_stack.DimensionOrganizationUID = dimension_organization_uid
    di_stack.DimensionIndexPointer = Tag(0x0020, 0x9057)    # InStackPositionNumber
    di_stack.FunctionalGroupPointer = Tag(0x0020, 0x9111)   # FrameContentSequence
    ds.DimensionIndexSequence = Sequence([di_stack])

    ds.PixelData = pixel_frames.tobytes()
    ds.save_as(str(fpath), enforce_file_format=True)


def read_dicom_quantity(quantity_dir, scale=None, offset=None):
    """Read one quantity written by save_4dflow_dicom: a directory of
    per-cardiac-phase 3D Enhanced MR volumes (phase000.dcm, phase001.dcm, …),
    each with NumberOfFrames = nz.

    If scale/offset are not given, they are taken from each file's
    RescaleSlope/RescaleIntercept (PixelValueTransformationSequence) – shared
    across all phases of a quantity.

    Returns
    -------
    np.ndarray, shape (nx, ny, nz, n_frames), dtype float32
    """
    import pydicom
    from pathlib import Path as _Path

    quantity_dir = _Path(quantity_dir)
    phase_files = sorted(
        quantity_dir.glob('phase*.dcm'),
        key=lambda p: int(re.search(r'phase(\d+)', p.stem).group(1)),
    )
    if not phase_files:
        raise FileNotFoundError(f"No phase*.dcm files found in {quantity_dir}")

    n_frames = len(phase_files)
    out = None
    for f, fp in enumerate(phase_files):
        ds  = pydicom.dcmread(str(fp))
        pix = ds.pixel_array                       # (nz, rows, cols)
        nx, ny, nz = ds.Columns, ds.Rows, ds.NumberOfFrames

        if scale is None or offset is None:
            pvt = ds.SharedFunctionalGroupsSequence[0].PixelValueTransformationSequence[0]
            sc, off = float(pvt.RescaleSlope), float(pvt.RescaleIntercept)
        else:
            sc, off = scale, offset

        if out is None:
            out = np.empty((nx, ny, nz, n_frames), dtype=np.float32)

        for s, item in enumerate(ds.PerFrameFunctionalGroupsSequence):
            s_idx = item.FrameContentSequence[0].InStackPositionNumber - 1
            out[:, :, s_idx, f] = _dequantize(pix[s].T, sc, off)

    return out


def save_4dflow_dicom(out_dir, dims, spacing, mask, signal, parameters, compressed=False):
    """Save 4D-flow MRI data as a study of single-stack ('3D') Enhanced MR
    DICOM volumes – one volume (own SeriesInstanceUID) per (quantity, cardiac
    phase), sharing StudyInstanceUID / FrameOfReferenceUID.

    See the module-level "DICOM I/O" comment block for the file layout,
    quantisation scheme (uint16 + RescaleSlope/Intercept via _quantize) and
    the LPS/RAF geometry note.

    Parameters
    ----------
    out_dir    : directory that will receive the per-quantity subdirectories
                 (created if absent)
    dims       : (nx, ny, nz)
    spacing    : (dx, dy, dz)  [m]  – used only as fallback when ijk_to_xyz /
                 xyz_to_raf are absent from parameters
    mask       : (*dims, n_frames)        uint8/bool
    signal     : (*dims, n_enc, n_frames) complex
    parameters : dict of acquisition metadata (kvs, vencs, enc_dirs,
                 ijk_to_xyz, xyz_to_raf, ms_parameters, …)
    compressed : True -> Deflated Explicit VR Little Endian (zlib),
                 False -> Explicit VR Little Endian (uncompressed)
    """
    import json
    import datetime
    from pathlib import Path as _Path
    from pydicom.uid import (ExplicitVRLittleEndian, DeflatedExplicitVRLittleEndian,
                             generate_uid)

    out_dir = _Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    transfer_syntax = DeflatedExplicitVRLittleEndian if compressed else ExplicitVRLittleEndian

    n_enc    = signal.shape[3]
    n_frames = signal.shape[4]

    ijk_to_lps = _ijk_to_lps(spacing, parameters)
    pixel_spacing, slice_thickness, orientation, positions = _dicom_geometry(ijk_to_lps, dims[2])

    # TR (ms) spaces the per-phase ContentTime/TriggerTime values
    tr_ms = _ms_parameters_dict(parameters).get('TR', 1.0)

    study_uid              = generate_uid()
    frame_of_reference_uid = generate_uid()
    base_time              = datetime.datetime.now()

    # ── Quantities to export, mirroring save_4dflow_nifti's file layout ──────
    quantities = [('mask', mask.astype(np.float32), 'OTHER', 'Mask')]
    for i in range(n_enc):
        quantities.append((f'magnitude_enc{i}', np.abs(signal[:, :, :, i, :]).astype(np.float32),
                           'MAGNITUDE', f'Magnitude (encoding {i})'))
    for i in range(n_enc):
        quantities.append((f'phase_enc{i}', np.angle(signal[:, :, :, i, :]).astype(np.float32),
                           'PHASE', f'Phase (encoding {i})'))
    if 'kvs' in parameters:
        from flowProcessing import extract_velocities
        kvs = np.asarray(parameters['kvs']).ravel().astype(np.float64)
        vel = extract_velocities(signal, kvs)            # (*dims, n_enc-1, n_frames)
        for i in range(vel.shape[3]):
            quantities.append((f'velocity_dir{i}', vel[:, :, :, i, :].astype(np.float32),
                               'OTHER', f'Velocity (direction {i})'))

    scales_offsets = {}
    for q_index, (name, data, cic, desc) in enumerate(quantities):
        q, scale, offset = _quantize(data)          # global scale/offset for all phases
        quantity_dir = out_dir / name
        quantity_dir.mkdir(parents=True, exist_ok=True)

        for f in range(n_frames):
            content_time = (base_time + datetime.timedelta(milliseconds=f * tr_ms)).strftime('%H%M%S.%f')
            _write_dicom_volume(
                quantity_dir / f'phase{f:03d}.dcm', q[:, :, :, f],
                pixel_spacing=pixel_spacing, slice_thickness=slice_thickness,
                orientation=orientation, positions=positions,
                scale=scale, offset=offset,
                series_uid=generate_uid(), series_number=q_index * n_frames + f + 1,
                series_description=f'{desc} (phase {f + 1}/{n_frames})',
                study_uid=study_uid, frame_of_reference_uid=frame_of_reference_uid,
                complex_image_component=cic, transfer_syntax=transfer_syntax,
                temporal_position_index=f + 1, n_temporal_positions=n_frames,
                content_time=content_time, trigger_time=f * tr_ms,
            )
        scales_offsets[name] = {'scale': scale, 'offset': offset}

    # ── Parameters JSON sidecar ───────────────────────────────────────────────
    params_out = {'dims': list(dims), 'spacing': list(spacing)}
    for key, val in parameters.items():
        try:
            params_out[key] = np.asarray(val).tolist()
        except Exception:
            pass
    params_out['scales_offsets'] = scales_offsets

    with open(out_dir / 'parameters.json', 'w') as fj:
        json.dump(params_out, fj, indent=2)


def load_4dflow_dicom(out_dir, compressed=False):
    """Load 4D-flow MRI data from the per-quantity directories of 3D Enhanced
    MR DICOM volumes written by save_4dflow_dicom.

    Reconstructs complex signal as magnitude * exp(i * phase).

    Returns
    -------
    dims       : tuple of int       (nx, ny, nz)
    spacing    : tuple of float     (dx, dy, dz)
    mask       : np.ndarray, shape (*dims, n_frames)        uint8
    signal     : np.ndarray, shape (*dims, n_enc, n_frames) complex64
    parameters : dict
    """
    import json
    from pathlib import Path as _Path

    out_dir = _Path(out_dir)

    parameters     = {}
    scales_offsets = {}
    params_path = out_dir / 'parameters.json'
    if params_path.exists():
        with open(params_path) as fj:
            raw = json.load(fj)
        scales_offsets = raw.pop('scales_offsets', {})
        for key, val in raw.items():
            try:
                parameters[key] = np.array(val)
            except Exception:
                pass

    dims    = tuple(int(v) for v in parameters['dims'])
    spacing = tuple(float(v) for v in parameters['spacing'])

    mag_dirs = sorted(
        (p for p in out_dir.glob('magnitude_enc*') if p.is_dir()),
        key=lambda p: int(re.search(r'enc(\d+)', p.name).group(1)),
    )
    if not mag_dirs:
        raise FileNotFoundError(f"No magnitude_enc* directories found in {out_dir}")

    n_enc  = len(mag_dirs)
    signal = None

    for i, mdir in enumerate(mag_dirs):
        so_mag   = scales_offsets.get(f'magnitude_enc{i}', {})
        mag      = read_dicom_quantity(mdir, so_mag.get('scale'), so_mag.get('offset'))
        so_phase = scales_offsets.get(f'phase_enc{i}', {})
        phase    = read_dicom_quantity(out_dir / f'phase_enc{i}',
                                       so_phase.get('scale'), so_phase.get('offset'))

        if signal is None:
            n_frames = mag.shape[3]
            signal   = np.empty((*dims, n_enc, n_frames), dtype=np.complex64)

        signal[..., i, :] = mag * (np.cos(phase) + 1j * np.sin(phase))

    so_mask = scales_offsets.get('mask', {})
    mask = read_dicom_quantity(out_dir / 'mask', so_mask.get('scale'), so_mask.get('offset'))
    mask = np.round(mask).astype(np.uint8)

    return dims, spacing, mask, signal, parameters
