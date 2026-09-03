"""
MUSE Reconstruction.

Author: Zhengguo Tan <zhengguo.tan@gmail.com>
"""

import numpy as np
import sigpy as sp

from sigpy import backend
from sigpy.mri import app, retro, sms
from sigpy.mri.dims import *


# %%
def _denoising(input, full_img_shape=None, use_iter=True, max_iter=5):
    """
    Args:
        input: acs images
        full_img_shape: shape of full-FOV images
    """

    # # Hanning
    if full_img_shape is None:
        full_img_shape = input.shape[-2:]

    device = backend.get_device(input)
    xp = device.xp

    with device:

        H = sp.hanning(input.shape[-2:], dtype=complex, symm=True,
                       device=device)

        H_full = sp.resize(H, full_img_shape)

        k_full = sp.resize(sp.fft(input, axes=[-2, -1]),
                           oshape=list(input.shape[:-2])
                           + list(full_img_shape))


        if use_iter:
            for m in range(max_iter):
                k_full = H_full * k_full
        else:
            k_full = H_full**max_iter * k_full


        img = sp.ifft(k_full, axes=[-2, -1])

        idx = abs(img) > 0
        phs = xp.zeros_like(img)
        phs[idx] = img[idx] / abs(img[idx])

    return img, phs


def sms_sense_linop(kdat, coils, yshift, 
                    phase_echo=None,
                    phase_shot=None,
                    weights=None,
                    real_value_constraint=False):

    device = backend.get_device(kdat)
    assert (device == backend.get_device(coils))

    Ncoil, Nz, Ny, Nx = coils.shape

    assert (Nz == len(yshift))

    phase_sms = sms.get_sms_phase_shift([Nz, Ny, Nx], Nz, yshift=yshift)

    img_shape = [1, Nz, Ny, Nx]

    if real_value_constraint is True:
        RVC = sp.linop.RealValueConstraint(img_shape)
    else:
        RVC = sp.linop.Identity(img_shape)

    if phase_echo is not None:

        ECO_PHS = sp.linop.Multiply(img_shape, 
                                    sp.to_device(phase_echo, device=device))

    else:

        ECO_PHS = sp.linop.Identity(img_shape)

    if phase_shot is not None:

        SEG = sp.linop.Multiply(ECO_PHS.oshape,
                                sp.to_device(phase_shot, device=device))

    else:

        SEG = sp.linop.Identity(ECO_PHS.oshape)

    # coils
    S = sp.linop.Multiply(SEG.oshape, coils)

    # FFT
    F = sp.linop.FFT(S.oshape, axes=range(-2, 0))

    # SMS
    SMS_PHS = sp.linop.Multiply(F.oshape, sp.to_device(phase_sms, device=device))
    SMS_SUM = sp.linop.Sum(SMS_PHS.oshape, axes=(DIM_Z, ), keepdims=True)

    SMS = SMS_SUM * SMS_PHS

    if weights is None:

        weights = app._estimate_weights(kdat, None, None, coil_dim=DIM_COIL)

    W = sp.linop.Multiply(SMS.oshape, weights**0.5)

    # sum over all echoes
    if phase_echo is not None:

        ECO_SUM = sp.linop.Sum(W.oshape, axes=(DIM_ECHO, ), keepdims=True)

    else:

        ECO_SUM = sp.linop.Identity(W.oshape)

    return ECO_SUM * W * SMS * F * S * SEG * ECO_PHS * RVC


def sms_sense_solve(A, y, lamda=0.01, tol=0, max_iter=30, verbose=False):

    device = backend.get_device((y))
    xp = device.xp

    AHA = lambda x: A.N(x) + lamda * x
    AHy = A.H(y)

    img = xp.zeros(A.ishape, dtype=y.dtype)
    alg_method = sp.alg.ConjugateGradient(AHA, AHy, img,
                            tol=tol,
                            max_iter=max_iter, verbose=verbose)

    while (not alg_method.done()):
        alg_method.update()

    return img

# %%
def calc_B0_phase(B0: np.ndarray, 
                  kdat: np.ndarray,
                  nonzero_ky_ind: np.ndarray,
                  ESP: float = 0.48, # ms
                  reverse_pe: bool = False):

    if B0.ndim == 2:
        B0 = B0[None, ...]

    N_z, N_y, N_x = B0.shape

    data_high_shape = nonzero_ky_ind.shape[:-1]
    data_high_len = np.prod(data_high_shape)

    ETL = nonzero_ky_ind.shape[-1]
    nonzero_ky_ind_flat = np.reshape(nonzero_ky_ind, [-1, ETL])

    weights = app._estimate_weights(kdat, None, None, coil_dim=-4)
    print('> weights shape: ', weights.shape)
    weight6 = np.reshape(weights, [data_high_len] + list(weights.shape[-4:]))

    eco_phase = []
    eco_weight = []

    for h in range(data_high_len):
        
        nonzero_ky_ind_h = nonzero_ky_ind_flat[h]

        print('> shot %2d'%(h), nonzero_ky_ind_h)

        for eco in range(ETL):

            eind = nonzero_ky_ind_h[eco]

            if (reverse_pe is True) and (h%2==1):

                delta_y = nonzero_ky_ind_h[-1] - eind
            
            else:

                delta_y = eind - nonzero_ky_ind_h[0]

            eco_phase.append( np.exp(-2j * np.pi * B0 * ESP * delta_y * 1e-3) )


            w = np.zeros_like(weights, shape=weights.shape[-4:])
            w[..., eind, :] = weight6[h, :, :, eind, :]

            eco_weight.append(w)

    eco_phase = np.array(eco_phase)
    eco_phase = np.reshape(eco_phase, list(data_high_shape) + [ETL, 1, N_z, N_y, N_x])

    eco_weight = np.array(eco_weight)
    eco_weight = np.reshape(eco_weight, list(data_high_shape) + [ETL, 1, N_z, N_y, N_x])

    return eco_phase, eco_weight


def _Resize(input: np.ndarray, oshape: list = [64, 64]):
    """
    Resize input (np.ndarray) to oshape using PyTorch interpolation.
    """
    import torchvision.transforms as T

    is_complex = np.iscomplexobj(input)

    input_tensor = sp.to_pytorch(input)

    TR = T.Resize(oshape, antialias=True)
    
    if is_complex is True:

        input_r = TR(input_tensor[..., 0]).cpu().detach().numpy()
        input_i = TR(input_tensor[..., 1]).cpu().detach().numpy()

        output = input_r + 1j * input_i

    else:
        output = TR(input_tensor).cpu().detach().numpy()

    return output

# %%
def MuseRecon(y: np.ndarray, coils: np.ndarray, 
              B0: np.ndarray = None,
              reverse_pe: bool = False,
              ESP: float = 0.48,
              MB: int = 1, 
              acs_shape: list = [64, 64],
              lamda: float = 0.001, 
              max_iter: int = 80, 
              tol: float = 0,
              use_readout_extend_fov: bool = False, 
              yshift: np.ndarray = None,
              real_value_constraint: bool = False,
              device: sp.Device = sp.cpu_device, 
              verbose: bool = False):
    """
    MUSE is a novel method to reconstruct one diffusion-weighted image (DWI)
    from multi-shot EPI acquisition. It consists of the following steps:
        1. shot-by-shot SENSE recon;
        2. phase estimation from every shot image;
        3. incorporate phase into phase-informed SENSE recon to obtain one DWI.

    Args:
        y (array): zero-filled k-space data with shape:
            [Nshot, Ncoil, Nz_collap, Ny, Nx], where
            - Nshot: # of shots per DWI,
            - Ncoil: # of coils,
            - Nz_collap: # of collapsed slices,
            - Ny: # of phase-encoding lines,
            - Nx: # of readout lines.

        coils (array): coil sensitivity maps with shape:
            [Ncoil, Nz, Ny, Nx], where
            - Nz: # of un-collapsed slices.

        B0 (array): B0 field map with shape:
            [Nz, Ny, Nx]

        ESP (float): echo spacing in ms.

        MB (int): multi-band factor
            MB = Nz / Nz_collap.

        acs_shape (tuple of ints): shape of the auto-calibration signal (ACS),
            which is used for the shot-by-shot SENSE recon.

    References:
        * Liu C, Moseley ME, Bammer R.
          Simultaneous phase correction and SENSE reconstruction for navigated multi-shot DWI with non-Cartesian k-space sampling.
          Magn Reson Med 2005;54:1412-1422.

        * Chen NK, Guidon A, Chang HC, Song AW.
          A robust multi-shot strategy for high-resolution diffusion weighted MRI enabled by multiplexed sensitivity-encoding (MUSE).
          NeuroImage 2013;72:41-47.

        * Dai E, Mani M, McNab JA.
          Multi-band multi-shot diffusion MRI reconstruction with joint usage of structured low-rank constraints and explicit phase mapping.
          Magn Reson Med 2023;89:95-111.
    """
    if y.ndim == 6:
        y = y[:, :, None, ...]

    Ndiff, Nshot, Necho, Ncoil, Nz_collap, Ny, Nx = y.shape
    assert(Nshot > 1)  # MUSE is a multi-shot technique

    _Ncoil, Nz, _Ny, _Nx = coils.shape

    assert ((Ncoil == _Ncoil) and (Ny == _Ny) and (Nx == _Nx))
    assert ((Nz_collap == Nz / MB))

    phi = sms.get_sms_phase_shift([MB, Ny, Nx], MB, yshift=yshift)

    # ACS data
    if acs_shape is None:

        ksp_acs = y.copy()
        mps_acs = coils.copy()

    else:

        ksp_acs = sp.resize(y, oshape=list(y.shape[:-2]) + list(acs_shape))

        # import torchvision.transforms as T
        # coils_tensor = sp.to_pytorch(coils)
        # TR = T.Resize(acs_shape, antialias=True)
        # mps_acs_r = TR(coils_tensor[..., 0]).cpu().detach().numpy()
        # mps_acs_i = TR(coils_tensor[..., 1]).cpu().detach().numpy()
        # mps_acs = mps_acs_r + 1j * mps_acs_i
        mps_acs = _Resize(coils, acs_shape)

    print('**** MUSE - ksp_acs shape ', ksp_acs.shape)
    print('**** MUSE - mps_acs shape ', mps_acs.shape)

    R_muse = []
    R_shot = []
    for z in range(Nz_collap):  # loop over collapsed k-space

        slice_idx = sms.get_uncollap_slice_idx(Nz, MB, z)
        mps_acs_slice = mps_acs[:, slice_idx, ...]

        for d in range(Ndiff):

            print('>> muse on slice ' + str(z).zfill(2) + ' diff ' + str(d).zfill(3))

            if use_readout_extend_fov:

                # 1. perform shot-by-shot ACS SENSE recon to estimate phase
                img_ext_shots = []
                for s in range(Nshot):  # loop over every shot

                    ksp = ksp_acs[d, s, :, z, ...]

                    ksp_ext, mps_ext, _ = sms.readout_extended_fov(ksp, mps_acs_slice, MB)

                    img_ext = app.SenseRecon(ksp_ext, mps_ext, 5E-5,
                                max_iter=90, tol=0,
                                device=device).run()

                    img_ext_shots.append(backend.to_device(img_ext))

                img_ext_shots = np.array(img_ext_shots)
                R_shot.append(img_ext_shots)

                # 2. phase estimation from shot images
                img_ext_shots_den, phs_ext_shots = _denoising(img_ext_shots, full_img_shape=[Ny, Nx * MB])

                img_ini = abs(np.mean(img_ext_shots_den * np.conj(phs_ext_shots), axis=0)).astype(phs_ext_shots.dtype)

                # 3. perform phase-informed SENSE recon
                # to estimate shot-combined DWI
                ksp = y[d, :, :, z, ...]
                mps = coils[:, slice_idx, ...]
                ksp_ext, mps_ext, _ = sms.readout_extended_fov(ksp, mps, MB)

                phs_ext_shots = np.expand_dims(phs_ext_shots, axis=1)
                phs_ext_shots_mps = phs_ext_shots * mps_ext
                phs_ext_shots_mps = phs_ext_shots_mps.reshape((-1, Ny, Nx * MB))

                # -- calculate weights for k-space
                arr1 = np.ones((Ncoil, Ny, Nx * MB))
                weights = (sp.rss(ksp_ext, axes=(1, ), keepdims=True) > 0.).astype(ksp_ext.dtype)
                weights = arr1 * weights

                ksp_ext = ksp_ext.reshape((-1, Ny, Nx * MB))
                weights = weights.reshape((-1, Ny, Nx * MB))

                img_ext = app.SenseRecon(ksp_ext, phs_ext_shots_mps,
                                         lamda, max_iter=max_iter, tol=tol,
                                         weights=weights,
                                         x=img_ini,
                                         device=device).run()

                img = sms.readout_unextend_fov(backend.to_device(img_ext), MB)
                # img = sp.ifft(np.conj(phi) * sp.fft(img, axes=[-2, -1]), axes=[-2, -1])
                R_muse.append(img)

            else:

                xp = device.xp

                eco_phase, eco_weight = None, None
                eco_phase_acs, eco_weight_acs = None, None

                if B0 is not None:

                    nonzero_ky_ind = retro.find_nonzero_ky_lines(y[d], ky_axis=-2)
                    eco_phase, eco_weight = calc_B0_phase(B0, y[d], nonzero_ky_ind, ESP, reverse_pe)
                    
                    if acs_shape is None:
                        B0_acs = B0.copy()
                        eco_phase_acs = eco_phase.copy()
                        eco_weight_acs = eco_weight.copy()

                    else:
                        B0_acs = _Resize(B0, acs_shape)
                        nonzero_ky_ind_acs = retro.find_nonzero_ky_lines(ksp_acs[d], ky_axis=-2)
                        eco_phase_acs, eco_weight_acs = calc_B0_phase(B0_acs, ksp_acs[d], nonzero_ky_ind_acs, ESP, reverse_pe)

                    eco_phase_acs_xp = sp.to_device(eco_phase_acs, device=device)
                    eco_weight_acs_xp = sp.to_device(eco_weight_acs, device=device)

                    eco_phase_xp = sp.to_device(eco_phase, device=device)
                    eco_weight_xp = sp.to_device(eco_weight, device=device)

                ksp_acs_xp = sp.to_device(ksp_acs, device=device)
                mps_acs_slice_xp = sp.to_device(mps_acs_slice, device=device)

                y_xp = sp.to_device(y, device=device)
                coils_xp = sp.to_device(coils, device=device)


                # 1. perform shot-by-shot ACS SENSE recon to estimate phase
                img_acs_shots = []
                for s in range(Nshot):

                    ksp = ksp_acs_xp[d, s, ..., z, :, :]
                    ksp = ksp[..., None, :, :]
                    print('> ksp: ', ksp.shape)

                    A = sms_sense_linop(ksp, mps_acs_slice_xp, yshift,
                                        phase_echo=eco_phase_acs_xp[s],
                                        weights=eco_weight_acs_xp[s])

                    img = sms_sense_solve(A, ksp, lamda=5E-5, tol=0,
                                          max_iter=max_iter,
                                          verbose=verbose)

                    img_acs_shots.append(sp.to_device(img[None, ...]))

                img_acs_shots = np.array(img_acs_shots)
                R_shot.append(img_acs_shots)

                # 2. phase estimation from shot images
                _, phs_shots = _denoising(img_acs_shots,
                                          full_img_shape=[Ny, Nx])

                # 3. perform phase-informed SENSE recon
                # to estimate shot-combined DWI
                ksp = y_xp[d, ..., z, :, :]
                ksp = ksp[..., None, :, :]
                mps = coils_xp[:, slice_idx, ...]

                A = sms_sense_linop(ksp, mps, yshift, 
                                    phase_shot=phs_shots,
                                    phase_echo=eco_phase_xp,
                                    weights=eco_weight_xp,
                                    real_value_constraint=real_value_constraint)

                img = sms_sense_solve(A, ksp, lamda=lamda, tol=tol,
                                      max_iter=max_iter, verbose=verbose)

                R_muse.append(sp.to_device(img))

    R_muse = np.array(R_muse)
    R_shot = np.array(R_shot)

    return R_muse, R_shot
