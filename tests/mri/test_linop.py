import unittest

import numpy as np
import numpy.testing as npt

import sigpy as sp
from sigpy.mri import linop, nlop

if __name__ == "__main__":
    unittest.main()


def check_linop_adjoint(A, dtype=float, device=sp.cpu_device):
    device = sp.Device(device)
    x = sp.randn(A.ishape, dtype=dtype, device=device)
    y = sp.randn(A.oshape, dtype=dtype, device=device)

    xp = device.xp
    with device:
        lhs = xp.vdot(A * x, y)
        rhs = xp.vdot(x, A.H * y)

        xp.testing.assert_allclose(lhs, rhs, atol=1e-5, rtol=1e-5)


class TestLinop(unittest.TestCase):
    def test_sense_model(self):
        img_shape = [16, 16]
        mps_shape = [8, 16, 16]

        img = sp.randn(img_shape, dtype=complex)
        mps = sp.randn(mps_shape, dtype=complex)

        A = linop.Sense(mps)

        check_linop_adjoint(A, dtype=complex)

        npt.assert_allclose(sp.fft(img * mps, axes=[-1, -2]), A * img)

    def test_sense_model_batch(self):
        img_shape = [16, 16]
        mps_shape = [8, 16, 16]

        img = sp.randn(img_shape, dtype=complex)
        mps = sp.randn(mps_shape, dtype=complex)

        for coil_batch_size in [None, 1, 2, 3]:
            A = linop.Sense(mps, coil_batch_size=coil_batch_size)
            check_linop_adjoint(A, dtype=complex)

            npt.assert_allclose(sp.fft(img * mps, axes=[-1, -2]),
                                A * img)

    def test_noncart_sense_model(self):
        img_shape = [16, 16]
        mps_shape = [8, 16, 16]

        img = sp.randn(img_shape, dtype=complex)
        mps = sp.randn(mps_shape, dtype=complex)

        y, x = np.mgrid[:16, :16]
        coord = np.stack([np.ravel(y - 8), np.ravel(x - 8)], axis=1)
        coord = coord.astype(float)

        A = linop.Sense(mps, coord=coord)
        check_linop_adjoint(A, dtype=complex)

        npt.assert_allclose(
            sp.fft(img * mps, axes=[-1, -2]).ravel(),
            (A * img).ravel(),
            atol=0.1,
            rtol=0.1,
        )

    def test_sense_tseg_off_res_model(self):
        img_shape = [16, 16]
        mps_shape = [8, 16, 16]

        img = sp.randn(img_shape, dtype=complex)
        mps = sp.randn(mps_shape, dtype=complex)

        y, x = np.mgrid[:16, :16]
        coord = np.stack([np.ravel(y - 8), np.ravel(x - 8)], axis=1)
        coord = coord.astype(float)

        d = np.sqrt(x * x + y * y)
        sigma, mu, a = 2, 0.25, 400
        b0 = a * np.exp(-((d - mu) ** 2 / (2.0 * sigma**2)))
        tseg = {"b0": b0, "dt": 4e-6, "lseg": 1, "n_bins": 10}

        F = sp.linop.NUFFT(mps_shape, coord)
        b, ct = sp.mri.util.tseg_off_res_b_ct(
            b0=b0, bins=10, lseg=1, dt=4e-6, T=coord.shape[0] * 4e-6
        )
        B1 = sp.linop.Multiply(F.oshape, b.T)
        Ct1 = sp.linop.Multiply(img_shape, ct.reshape(img_shape))
        S = sp.linop.Multiply(img_shape, mps)

        A = linop.Sense(mps, coord=coord, tseg=tseg)

        check_linop_adjoint(A, dtype=complex)
        npt.assert_allclose(B1 * F * S * Ct1 * img, A * img)

    def test_noncart_sense_model_batch(self):
        img_shape = [16, 16]
        mps_shape = [8, 16, 16]

        img = sp.randn(img_shape, dtype=complex)
        mps = sp.randn(mps_shape, dtype=complex)

        y, x = np.mgrid[:16, :16]
        coord = np.stack([np.ravel(y - 8), np.ravel(x - 8)], axis=1)
        coord = coord.astype(float)

        for coil_batch_size in [None, 1, 2, 3]:
            A = linop.Sense(mps, coord=coord, coil_batch_size=coil_batch_size)
            check_linop_adjoint(A, dtype=complex)

            npt.assert_allclose(
                sp.fft(img * mps, axes=[-1, -2]).ravel(),
                (A * img).ravel(),
                atol=0.1,
                rtol=0.1,
            )

    if sp.config.mpi4py_enabled:

        def test_sense_model_with_comm(self):
            img_shape = [16, 16]
            mps_shape = [8, 16, 16]
            comm = sp.Communicator()

            img = sp.randn(img_shape, dtype=complex)
            mps = sp.randn(mps_shape, dtype=complex)
            comm.allreduce(img)
            comm.allreduce(mps)
            ksp = sp.fft(img * mps, axes=[-1, -2])

            A = linop.Sense(mps[comm.rank :: comm.size], comm=comm)

            npt.assert_allclose(
                A.H(ksp[comm.rank :: comm.size]),
                np.sum(sp.ifft(ksp, axes=[-1, -2]) * mps.conjugate(), 0),
            )

    def test_sense_subspace_model(self):
        basis_shape = [80, 5]
        img_shape = [5, 1, 16, 16]
        mps_shape = [8, 16, 16]

        basis = sp.randn(basis_shape, dtype=complex)
        img = sp.randn(img_shape, dtype=complex)
        mps = sp.randn(mps_shape, dtype=complex)

        A = linop.Sense(mps, basis=basis)

        check_linop_adjoint(A, dtype=complex)

        full_img = basis @ np.reshape(img, (5, -1))
        full_img = np.reshape(full_img, [80] + img_shape[1:])

        npt.assert_allclose(sp.fft(full_img * mps, axes=[-1, -2]),
                            A * img)

    def test_water_fat_model(self):

        TE = np.array([2.46, 3.69, 4.92, 6.15, 7.38]) * 1e-3
        zm = nlop.calc_fat_modu(TE, B0=3.0)
        N_echo, N_param = zm.shape

        input_shape = [5, 1, N_param, 1, 1, 16, 16]
        input = sp.randn(input_shape, dtype=complex)

        output_shape = [5, 1, N_echo, 1, 1, 16, 16]

        # direct computation
        wf0 = np.zeros(output_shape, dtype=complex)
        w = input[:, :, 0, ...]
        f = input[:, :, 1, ...]
        for e in range(N_echo):
            wf0[:, :, e, ...] = zm[e, 0] * w + zm[e, 1] * f

        # computation using linear operator
        T1 = sp.linop.Transpose(input_shape, [-5, -6, -7, -4, -3, -2, -1])
        R1 = sp.linop.Reshape([T1.oshape[0], np.prod(T1.oshape[1:])], T1.oshape)
        MM = sp.linop.MatMul(R1.oshape, zm)
        R2 = sp.linop.Reshape([N_echo] + list(T1.oshape[1:]), MM.oshape)
        T2 = sp.linop.Transpose(R2.oshape, [-5, -6, -7, -4, -3, -2, -1])
        A = T2 * R2 * MM * R1 * T1
        # print('> A ishape: ', A.ishape, ', oshape: ', A.oshape)
        wf1 = A * input

        npt.assert_allclose(wf0, wf1)