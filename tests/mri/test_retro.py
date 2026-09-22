import unittest

import numpy as np
import numpy.testing as npt
import random

import sigpy as sp

from sigpy.mri import retro
from sigpy.mri.dims import *

if __name__ == "__main__":
    unittest.main()

class TestApp(unittest.TestCase):
    def test_find_nonzero_ky_lines(self):
        kdat = np.zeros([1, 1, 40, 1, 1, 150, 100])

        N_h = np.prod(kdat.shape[:DIM_ECHO])
        N_y = kdat.shape[DIM_Y]
        ETL = 35

        kdat6 = np.reshape(kdat, [-1] + list(kdat.shape[DIM_ECHO:]))

        ky_ind_i = []
        for h in range(N_h):
            ky_ind = random.sample(range(N_y), ETL)
            ky_ind = np.sort(ky_ind)

            kdat6[h, ..., ky_ind, :] = 1.
            ky_ind_i.append(ky_ind)

        ky_ind_i = np.array(ky_ind_i)
        ky_ind_i = np.reshape(ky_ind_i, list(kdat.shape[:DIM_ECHO]) + [-1])

        ky_ind_o = retro.find_nonzero_ky_lines(kdat, ky_axis=DIM_Y)

        npt.assert_allclose(ky_ind_i, ky_ind_o)