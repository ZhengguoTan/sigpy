import unittest
import numpy.testing as npt
from sigpy import backend, config, fourier, linop, util

from sigpy.mri import nlop

import sigpy as sp

if __name__ == '__main__':
    unittest.main()

devices = [backend.cpu_device]
if config.cupy_enabled:
    devices.append(backend.Device(0))


class TestNlop(unittest.TestCase):

    def check_nlop_derivative(self, A,
                              device=backend.cpu_device,
                              dtype=float):
        device = backend.Device(device)

        scale = 1e-8
        x = util.randn(A.ishape, dtype=dtype, device=device)
        h = util.randn(A.ishape, dtype=dtype, device=device)

        with device:
            dy1 = (A.forward(x + scale * h) - A.forward(x)) / scale
            dy2 = A.derivative(x, h)

            npt.assert_allclose(backend.to_device(dy1),
                                backend.to_device(dy2),
                                atol=1e-5, rtol=1e-5,
                                err_msg=A.repr_str + ' derivative operator!')

    def check_nlop_adjoint(self, A, device=backend.cpu_device, dtype=float):
        device = backend.Device(device)
        x = util.randn(A.ishape, dtype=dtype, device=device)

        dx = util.randn(A.ishape, dtype=dtype, device=device)
        dy = util.randn(A.oshape, dtype=dtype, device=device)

        xp = device.xp
        with device:
            lhs = xp.vdot(A.derivative(x, dx), dy)
            rhs = xp.vdot(dx, A.adjoint(x, dy))

            npt.assert_allclose(backend.to_device(lhs),
                                backend.to_device(rhs),
                                atol=1e-5, rtol=1e-5,
                                err_msg=A.repr_str + ' adjoint operator!')

    def test_nlinv_model(self):
        for device in devices:
            xp = device.xp

            I = util.randn((1, 3, 3), dtype=complex, device=device)
            C = util.randn((4, 3, 3), dtype=complex, device=device)

            A = nlop.Nlinv(I.shape, C.shape, W_coil=False)

            x = device.xp.ones(A.ishape, dtype=complex)
            x[0, :, :] = I
            x[1:, :, :] = C

            F = linop.FFT(C.shape, axes=(-2, -1))

            # test forward
            y1 = F(I*C)
            y2 = A.forward(x)

            npt.assert_allclose(backend.to_device(y2),
                                backend.to_device(y1),
                                err_msg='forward operator!')

            # test derivative
            dx = util.randn(x.shape, dtype=complex, device=device)

            dy1 = F(dx[0, :, :] * C + I * dx[1:, :, :])
            dy2 = A.derivative(x, dx)

            npt.assert_allclose(backend.to_device(dy2),
                                backend.to_device(dy1),
                                err_msg='derivative operator!')

            # test adjoint
            dx1 = xp.zeros(x.shape, dtype=complex)

            dI1 = xp.sum(xp.conj(C) * F.H(dy1), axis=0)
            dC1 = xp.conj(I) * F.H(dy1)

            dx1[0, :, :] = dI1
            dx1[1:, :, :] = dC1

            dx2 = A.adjoint(x, dy2)

            npt.assert_allclose(backend.to_device(dx2),
                                backend.to_device(dx1),
                                err_msg='adjoint operator!')

    def test_diffusion_model(self):
        for device in devices:
            xp = device.xp

            # Exponential
            tvec = util.randn((15, 6), dtype=float, device=device)
            b0 = util.randn((1, 1, 3, 3), dtype=complex, device=device)
            D = util.randn((6, 1, 3, 3), dtype=complex, device=device)
            x = xp.concatenate((b0, D))

            Dr = xp.reshape(D, (D.shape[0], -1))
            y1 = xp.exp(xp.matmul(tvec, Dr))
            y1 = xp.reshape(y1, (15, 1, 3, 3))
            y1 *= b0

            # Sense
            coils = util.randn((8, 3, 3), dtype=complex, device=device)

            y2 = fourier.fft(coils * y1, axes=(-2, -1))

            A = nlop.Diffusion(x.shape, tvec, coils)
            y = A(x)

            npt.assert_allclose(backend.to_device(y),
                                backend.to_device(y2),
                                err_msg='Diffusion model mismatch!')

    def test_WFZ_model(self):
        for device in devices:
            xp = device.xp

            # Exponential
            tvec = util.randn((5, 1), dtype=float, device=device)
            N_echo = tvec.shape[0]

            Wat = util.randn((1, 1, 3, 3), dtype=complex, device=device)
            Fat = util.randn((1, 1, 3, 3), dtype=complex, device=device)
            fB0 = util.randn((1, 1, 3, 3), dtype=complex, device=device)
            x = xp.concatenate((Wat, Fat, fB0))

            Model = nlop.WFZ(x.shape, tvec, device=device)
            y1 = Model(x)

            # test forward
            fm = nlop.calc_fat_modu(tvec * 1E-3)
            y2 = xp.zeros(Model.oshape, dtype=complex)

            for e in range(tvec.shape[0]):
                y2[e] = (fm[e, 0] * Wat + fm[e, 1] * Fat) *\
                    xp.exp(2j * xp.pi * fB0 * tvec[e, 0])

            npt.assert_allclose(backend.to_device(y1), 
                                backend.to_device(y2), 
                                err_msg='WFZ model mismatch!')

            # test derivative
            self.check_nlop_derivative(Model, device=device, dtype=x.dtype)

            # test adjoint
            self.check_nlop_adjoint(Model, device=device, dtype=x.dtype)


    # def test_MGRE_model(self):
    #     for device in devices:
    #         xp = device.xp

    #         # Exponential
    #         tvec = util.randn((5, 1), dtype=float, device=device)
    #         M0 = util.randn((1, 1, 3, 3), dtype=complex, device=device)
    #         B0 = util.randn((1, 1, 3, 3), dtype=complex, device=device)
    #         x = xp.concatenate((M0, B0))

    #         Dr = xp.reshape(B0, (B0.shape[0], -1))
    #         y1 = xp.exp(xp.matmul(tvec, Dr))
    #         y1 = xp.reshape(y1, (5, 1, 3, 3))
    #         y1 *= M0

    #         # Sense
    #         coils = util.randn((8, 3, 3), dtype=complex, device=device)

    #         y2 = fourier.fft(coils * y1, axes=(-2, -1))

    #         A = nlop.MGRE(x.shape, tvec, coils)
    #         y = A(x)

    #         npt.assert_allclose(backend.to_device(y),
    #                             backend.to_device(y2),
    #                             err_msg='B0Field model mismatch!')