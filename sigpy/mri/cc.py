"""Coil compression functions.

Author: Zhengguo Tan <zhengguo.tan@gmail.com>
"""
import numpy as np
import sigpy as sp

from scipy.linalg import eig

__all__ = ['scc', 'gcc', 'rovir']


# %%
def scc(kdat, P=10, coil_dim=-2, device=sp.cpu_device):
    r"""Coil compression based on SVD.

    Args:
        kdat (array): raw k-space data.
        P (int): number of virtual coils to be kept. [Default: 10].
        coil_dim (int): the coil dimension. [Default: -2].
        device: use CPU or GPU device.

    Returns:
        coil compressed k-space data, and
        truncated eigen vectors.

    References:
        * Buehrer M., Pruessmann K. P., Boesiger P., Kozerke S. (2007).
          Array compression for MRI with large coil arrays.
          Magn. Reson. Med., 2007.

        * Huang F., Vijayakumar S., Li Y., Hertel S., Duensing G. R. (2008).
          A software channel compression technique for faster reconstruction with many channels.
          Magn. Reson. Imaging., 26, 133-141.
    """
    if P >= kdat.shape[coil_dim]:
        print('> return the original data')
        return kdat, None

    device = sp.Device(device)
    xp = device.xp

    with device:
        # move the dimension of coils to 0
        y1 = xp.swapaxes(sp.to_device(kdat, device=device), coil_dim, 0)
        y2 = xp.reshape(y1, (y1.shape[0], -1))

        # covariance matrix: [num_coil, num_coil]
        yc = xp.cov(y2)

        eigvals, eigvecs = xp.linalg.eigh(yc)

        # eigvals and eigvecs in descending order
        idx = eigvals.argsort()[::-1]
        eigvals = eigvals[idx]
        eigvecs = eigvecs[:, idx]

        print('> energy kept: ' +
              '%3.4f'%(xp.sum(eigvals[:P]) / xp.sum(eigvals)))

        S = eigvecs[:, :P]

        y3 = xp.conj(S.T) @ y2

        y4 = xp.reshape(y3, [P] + list(y1.shape[1:]))
        y5 = xp.swapaxes(y4, coil_dim, 0)

        return sp.to_device(y5, device=sp.get_device(kdat)), S

# %%
# TODO: geometric coil compression
def gcc():
    None

def gram_schmidt(A):
    r'''

    Args:
        input: A set of linearly independent vectors stored
              as the columns of matrix A
    Return:
        outpt: An orthongonal basis for the column space of A.

    Ref:
        https://www.sfu.ca/~jtmulhol/py4math/linalg/np-gramschmidt/
    '''
    # get the number of vectors.
    A = np.copy(A)  # -> .astype(np.float64) # create a local instance of the array
    n = A.shape[1]
    for j in range(n):
        # For the vector in column j, find the perpendicular
        # of the projection onto the previous orthogonal vectors.
        for k in range(j):
            A[:, j] -= np.dot(A[:, k], A[:, j]) * A[:, k]
        # If original vectors aren't lin indep then we can check for this:
        # (1)
        if np.isclose(np.linalg.norm(A[:, j]), 0, rtol=1e-15, atol=1e-14, equal_nan=False):
            A[:, j] = np.zeros(A.shape[0])
        else:
            A[:, j] = A[:, j] / np.linalg.norm(A[:, j])
    return A

# %%
def rovir(kdat: np.ndarray,
          coil_images: np.ndarray,
          thresh: float = 0.60,
          baseres: int = 192,
          overgrid: float = 2.0,
          roi_signal: np.ndarray = None,
          roi_interf: np.ndarray = None):
    r"""Coil reduction based on ROVIR.
    Args:
        kdat (array): raw k-space data.
        coil_images (array): time-averaged coil images.
        baseres (int): base resolution. [Default: 192].
        overgrid (float): overgrid factor. [Default: 2.0].
        roi_signal (array): region of signal. [Default: None].
        roi_interf (array): region of interference. [Default: None].

    Returns:
        kdat_v (array): kdat with reduced number of coils.

    References:
        * Kim D., Cauley S. F., Nayak K. S., Leahy R. M., Halder J. P. (2021).
          Region-optimized virtual (ROVir) coils: Localization and/or suppression of spatial regions using sensor-domain beamforming.
          Magn. Reson. Med., 2021.
    """
    N_G = int(baseres * overgrid)

    N_time, _, N_arm, N_readout = kdat.shape

    N_C, N_Y, N_X = coil_images.shape

    assert N_C == kdat.shape[-3]
    assert N_Y == N_G
    assert N_X == N_G

    if roi_signal is None:
        roi_signal = np.zeros((N_G, N_G), dtype=np.float32)
        ind0 = int((N_G - baseres)//2)
        ind1 = ind0 + baseres
        roi_signal[ind0:ind1, ind0:ind1] = 1

    if roi_interf is None:
        roi_interf = np.zeros((N_G, N_G), dtype=np.float32)
        buffer = int(N_G * 1.25 / 2)
        ind0 = int((N_G - buffer)//2)
        ind1 = ind0 + buffer
        roi_interf[ind0:ind1, ind0:ind1] = 1
        roi_interf = 1 - roi_interf


    Xs = np.reshape(roi_signal * coil_images, [N_C, -1])
    Xi = np.reshape(roi_interf * coil_images, [N_C, -1])

    A = Xs @ Xs.conj().T  # [N_C, N_C]
    B = Xi @ Xi.conj().T  # [N_C, N_C]

    N_V = int(N_C * thresh)
    w, vl, vr = eig(A, b=B, left=True, right=True)
    vr_gs, _ = np.linalg.qr(vr) # gram_schmidt(vr)
    vr_keep = vr_gs[:, :N_V].T

    # [N_V, N_C, N_time, N_arm, N_readout]
    weights = np.tile(vr_keep[..., None, None, None],
                      [1, 1, N_time, N_arm, N_readout])
    # [N_V, N_time, N_C, N_arm, N_readout]
    weights = np.transpose(weights, axes=(0, 2, 1, 3, 4))

    kdat_v = np.sum(weights * kdat, axis=-3)
    kdat_v = np.swapaxes(kdat_v, 0, 1)

    return kdat_v
