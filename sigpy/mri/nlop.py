# -*- coding: utf-8 -*-
"""MRI non-linear operators.

This module contains these non-linear operators:

    * Nlinv,
        joint coil sensitivity maps and image content.

    * Diffusion,
        exponential diffusion modelling and parallel imaging sampling.

    * MGRE,
        multi gradient echoes.

Author:
    * 2022-2026 Zhengguo Tan <zhengguo.tan@gmail.com>
"""
import math
import numpy as np
import sigpy as sp

from sigpy import backend, nlop
from sigpy.mri import linop

from typing import List, Literal

# %%
class Nlinv(sp.nlop.Nlop):
    """
    Construction of the non-linear parallel imaging (nlinv) operator.

    Given the unknown x = (rho, c_1, ..., c_N)^T
    , where
        rho: image content, and
        c_1, ..., c_N: N coil sensitivity maps,

    the forward operation is,

        F(x) = ( ..., FT{rho * c_n}, ... )^T

    , where
        n in [1, N], and
        FT is either masked FFT or NUFFT.

    Args:
        image_shape (tuple): shape of image.
        coil_shape (tuple): shape of coils.
        coord (None or array): coordinates, i.e. trajectories
        coil (None or array): coil sensitivity maps.
        W_coil (boolean): apply Sobolev weight on coil or not.
        upd_coil (boolean): update coil sensitivity maps or not.

    Reference:
        Bauer F., Kannengiesser S. (2007).
        An alternative approach to the image reconstruction
        for parallel data acquisition in MRI.
        Math. Methods Appl. Sci., 30, 1437-1451.

        Uecker M., Hohage T., Block K. T., Frahm J. (2008).
        Image reconstruction by regularized nonlinear inversion -
        joint estimation of coil sensitivities and image content.
        Magn. Reson. Med., 60, 674-682.
    """

    def __init__(self, image_shape, coil_shape,
                 coord=None, coil=None,
                 W_coil=True, upd_coil=True,
                 repr_str=None):
        self.image_shape = image_shape
        self.coil_shape = coil_shape

        ishape = self._get_xshape()

        self.coord = coord
        self.coil = coil
        self.upd_coil = upd_coil

        # Sobolev linear operator on coils
        if W_coil:
            self.W = sp.linop.Sobolev(self.coil_shape)
        else:
            self.W = sp.linop.Identity(self.coil_shape)

        # FFT or NUFFT operator
        x_ndim = len(ishape)
        if coord is None:
            self.F = sp.linop.FFT(self.coil_shape, axes=range(-x_ndim+1, 0))
        else:
            self.F = sp.linop.NUFFT(self.coil_shape, coord)

        oshape = self.F.oshape

        super().__init__(oshape, ishape, repr_str)

    def _get_xshape(self):

        image_ndim = len(self.image_shape)

        if image_ndim == 2:
            num_coilimg = 1 + self.coil_shape[0]
        else:
            num_coilimg = self.image_shape[0] + self.coil_shape[0]

        xshape = []   # empty list
        xshape.append(num_coilimg)

        return xshape + list(self.image_shape[-2:])

    def _forward(self, input):
        with sp.backend.get_device(input):

            # store the current estimate into class
            self.x = input

            image = self.x[0, :, :]   # extract image
            coil_ksp = self.x[1:, :, :]   # extract coils
            coil_img = self.W * coil_ksp

            return self.F(image * coil_img)

    def _get_Jacobian(self, x):
        return None

    def _derivative(self, x, dx):
        device = sp.backend.get_device(dx)

        self.x = x

        with device:
            image = self.x[0, :, :]
            coil_ksp = self.x[1:, :, :]
            coil_img = self.W * coil_ksp

            dimage = dx[0, :, :]
            dcoil_ksp = dx[1:, :, :]
            dcoil_img = self.W * dcoil_ksp

            return self.F * (dimage * coil_img + image * dcoil_img)

    def _adjoint(self, x, dy):
        device = sp.backend.get_device(dy)
        xp = device.xp

        self.x = x

        output = xp.zeros_like(self.x)

        with device:
            image = self.x[0, :, :]
            coil_ksp = self.x[1:, :, :]
            coil_img = self.W * coil_ksp

            dcoilimg = self.F.H * dy

            output[0, :, :] = xp.sum(xp.conj(coil_img) * dcoilimg, axis=0)

            if self.upd_coil:
                output[1:, :, :] = self.W.H(xp.conj(image) * dcoilimg)

            return output


# %% class Diffusion(sp.nlop.Nlop):
def Diffusion(input_shape, diff_enc, coil,
              scale=None, rvc=False, dwi_phase=None,
              coord=None, weights=None):
    """
    Construction of the non-linear Diffusion operator.

    Given the unknown x = D
    , where
        D: diffusion tensor,

    the forward operation is,

        A(x) = E * P, and E = F * S

    , where
        E (linop): the SENSE linear operator,
        F (linop): k-space sampling operator,
        S (linop): multiply with coil sensitivity maps, and
        P (nlop): exponential diffusion model.

    Args:
        input_shape (tuple): shape of input images.
        diff_enc (array): diffusion encoding matrix,
        i.e. the output matrix from sp.mri.epi.get_B().
        coil (array): coil sensitivity maps.
        coord (None or array): coordinates, i.e. trajectories.
    """
    const_b0 = True if input_shape[0] == 6 or input_shape[0] == 21 else False

    P = nlop.Exponential(input_shape, diff_enc,
                         const_a=const_b0, rvc=rvc, scale=scale)

    # phase correction for every diffusion-weighted image
    if dwi_phase is not None:
        I = sp.linop.Multiply(P.oshape, dwi_phase)
    else:
        I = sp.linop.Identity(P.oshape)

    # parallel imaging forward model (Sense)
    E = linop.Sense(coil, ishape=P.oshape, coord=coord, weights=weights)

    # Compose (i.e. Chain) Sense linop with Diffusion nlop
    A = E * I * P
    A.repr_str = 'Diffusion'

    return A

# %%
def calc_fat_modu(TE,  # second
                  GYRO: float = 42.57747892E6,  # Hz/T
                  B0: float = 0.55  # T
                  ):
    """
    calculate fat modulation

    Input:
        TE - echo time in second
        GYRO - Gyromagnetic ratio
        B0 - Magnetic field strength
    """
    device = backend.get_device(TE)
    xp = device.xp

    with device:

        TE = TE.ravel()

        ppm = xp.array([-3.80, -3.40, -2.60, -1.94, -0.39, +0.60]) * 1E-6
        # amp = xp.array([0.087, 0.693, 0.128, 0.004, 0.039, 0.048])
        amp = xp.array([0.086, 0.537, 0.165, 0.046, 0.052, 0.114])

        res = xp.zeros([len(TE), 2], dtype=complex)

        for n in range(len(TE)):

            for p in range(len(ppm)):

                res[n, 1] += amp[p] * xp.exp(2j * math.pi * GYRO * B0 * ppm[p] * TE[n])

        res[:, 0] = 1.

    return res

class WFZ(sp.nlop.Nlop):
    """
    Construction of the non-linear operator WFZ

    Given the unknown x = (W, F, Z)^T, where
        W: water
        F: fat
        Z: fB0 * 2 * pi + R2*
    and the encoding array encode (usually the echo times t),

    the forward operation is
        F_t (x) = (W + F * fm_t) * exp(encode_t * Z)

    fm_t is the fat-modulated phases along echo times

    Args:
        ishape (tuple): input shape
        encode (array): echo times in milli-second (ms)
        scale (list): TODO

    References:
        Tan Z., Unterberg-Buchwald C., Blumenthal M., et al. (2023).
        Free-breathin liver fat, R2* and B0 field mapping
        using multi-echo radial FLASH and
        regularized model-based reconstruction.
        IEEE Trans. Med. Imaging, 42, 1374-1387.
    """
    def __init__(self, ishape, encode,
                 B0: float = 0.55,  # Tesla
                 scale: List[float] = None,
                 rvc: List[bool] = None,
                 device=backend.Device(-1),
                 repr_str: str = None):
        image_shape = list(ishape[1:])
        self.num_param = ishape[0]

        # encode: [num_echo, 1]
        self.num_echo = encode.shape[0]
        assert encode.shape[1] == 1

        oshape = [self.num_echo] + image_shape

        self.encode = backend.to_device(encode, device)
        self.device = device

        self.fm = calc_fat_modu(self.encode * 1E-3, B0=B0)  # second

        print('   > fm:')
        print(self.fm)


        if scale is None:
            scale = [1.] * self.num_param
        else:
            assert self.num_param == len(scale)

        if rvc is None:
            self.rvc = [False] * self.num_param
        else:
            if len(rvc) == 1:
                self.rvc = list(rvc) * self.num_param
            else:
                self.rvc = list(rvc)
                assert self.num_param == len(self.rvc)

        super().__init__(oshape, ishape, scale=scale, repr_str=repr_str)

    def _get_params(self, x):

        assert backend.get_device(x) == self.device

        with self.device:

            Wat = x[0, ...]
            Fat = x[1, ...]
            fB0 = x[2, ...]

        return Wat, Fat, fB0

    def _forward(self, input):

        assert self.device == backend.get_device(input)
        assert self.num_param == input.shape[0]

        xp = self.device.xp
        with self.device:

            self.x = input

            for p in range(self.num_param):
                self.x[p] *= self.scale[p]

            Wat, Fat, fB0 = self._get_params(self.x)

            output = xp.zeros(self.oshape, dtype=complex)
            for e in range(self.num_echo):

                output[e] = (self.fm[e, 0] * Wat + self.fm[e, 1] * Fat) *\
                    xp.exp(2j * xp.pi * fB0 * self.encode[e, 0])

        return output

    def _get_Jacobian(self, x):

        assert self.device == backend.get_device(x)

        Jshape = [self.num_echo] + list(self.x.shape)

        xp = self.device.xp
        with self.device:

            self.x = x
            Wat, Fat, fB0 = self._get_params(self.x)

            output = xp.zeros(Jshape, dtype=complex)

            for e in range(self.num_echo):
                phase_fB0 = xp.exp(2j * xp.pi * fB0 * self.encode[e, 0])

                # Jacobian for Water
                output[e, 0, ...] = phase_fB0

                # Jacobian for Fat
                output[e, 1, ...] = self.fm[e, 1] * phase_fB0

                # Jacobian for fB0
                output[e, 2, ...] = (self.fm[e, 0] * Wat + self.fm[e, 1] * Fat) *\
                    phase_fB0 * (2j * xp.pi * self.encode[e, 0])

        return output

    def _derivative(self, x, dx):

        assert self.device == backend.get_device(x)
        assert self.device == backend.get_device(dx)

        xp = self.device.xp
        with self.device:
            self.Jacobian = self._get_Jacobian(x)
            # print('   > Jacobian shape: ', self.Jacobian.shape)
            return xp.sum(self.Jacobian * dx, axis=1)

    def _adjoint(self, x, dy):

        assert self.device == backend.get_device(x)
        assert self.device == backend.get_device(dy)

        xp = self.device.xp
        with self.device:
            self.Jacobian = self._get_Jacobian(x)
            JH = xp.conjugate(xp.moveaxis(self.Jacobian, 0, 1))
            dx = xp.sum(JH * dy, axis=1)

            for e in range(self.num_param):
                if self.rvc[e] is True:
                    dx[e, ...] = dx[e, ...].real + 0. * 1j

            return dx


def MGRE(input_shape, TE, coil,
         model_waterfat: bool = False,
         B0: float = 0.55,
         trafos: Literal['sobolev', 'identity'] = 'identity',
         scale=None, rvc=False,
         coord=None, weights=None):
    """
    Model Multi-Echo Gradient Echo Signal
    """
    device = sp.backend.get_device(coil)
    TE = sp.to_device(TE, device=device)

    N_param = 3 if model_waterfat is True else 2
    assert N_param == input_shape[0]

    if trafos == 'identity':

        single_shape = [1] + list(input_shape[1:])

        scaling_ops = []
        for p in range(N_param):
            scaling = 1  # if p < N_param-1 else 1E-3
            scaling_ops.append(sp.linop.Multiply(single_shape, scaling))

        T = sp.linop.Diag(scaling_ops, iaxis=0, oaxis=0)

    elif trafos == 'sobolev':

        excl_fB0_shape = [N_param-1] + list(input_shape[1:])
        incl_fB0_shape = [1] + list(input_shape[1:])

        op1 = sp.linop.Identity(excl_fB0_shape)
        op2 = sp.linop.Sobolev(incl_fB0_shape, a=11, b=2)

        T = sp.linop.Diag([op1, op2], iaxis=0, oaxis=0)

    print(' -> T ishape: ', T.ishape, ' oshape: ', T.oshape)


    if model_waterfat is False:

        P = nlop.Exponential(input_shape, TE,
                             const_a=False, rvc=rvc, scale=scale)

    else:

        P = WFZ(input_shape, TE, scale=scale, rvc=rvc, device=device)

    print(' -> P ishape: ', P.ishape, ' oshape: ', P.oshape)

    # Parallel Imaging
    E = linop.Sense(coil, ishape=P.oshape, coord=coord, weights=weights)

    print(' -> E ishape: ', E.ishape, ' oshape: ', E.oshape)

    # Compose (i.e. Chain) Sense linop with B0 nlop
    A = E * P * T
    A.repr_str = 'MGRE'
    A.trafos = T

    return A
