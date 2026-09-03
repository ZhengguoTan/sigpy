# -*- coding: utf-8 -*-
"""Methods for Echo-Planar Imaging (EPI) acquisition:

* retrospectively undersampling phase-encoding direction
* split shots within one diffusion encoding
* find non-zero ky lines from Cartesian k-space data

Author:
    Zhengguo Tan <zhengguo.tan@gmail.com>
"""
import numpy as np

from .dims import *

# %%
def unsamp_ky(kdat: np.ndarray, 
              pe_axis: int = -3,
              uniform_unsamp: bool = True, 
              unsamp_factor: int = 2):
    """retrospectively undersample phase-encoding direction
    Input:
        kdat: k-space data, shape [..., ky, ...]
        pe_axis: phase-encoding axis [default: -3]
        uniform_unsamp: whether to uniformly undersample [default: True]
        unsamp_factor: undersampling factor [default: 2]
    
    Output:
        output: retrospectively undersampled k-space data, shape [..., ky, ...]
    """
    # find valid phase-encoding lines
    kdat1 = np.swapaxes(kdat, pe_axis, 0)
    kdat2 = np.reshape(kdat1, (kdat1.shape[0], -1))
    kdat3 = np.sum(kdat2, axis=1)

    sampled_phaenc_ind = np.array(np.nonzero(kdat3)).ravel()
    sampled_phaenc_len = len(sampled_phaenc_ind)

    loop_shape = [np.prod(kdat.shape[:pe_axis])] \
        + list(kdat.shape[pe_axis:])
    kdat4 = np.reshape(kdat, loop_shape)

    output = np.zeros_like(kdat4)

    shift_cnt = 0

    # loop over all high dimensions
    for l in range(loop_shape[0]):
        if uniform_unsamp:
            rinds = np.arange(shift_cnt, sampled_phaenc_len, unsamp_factor)
            shift_cnt = (shift_cnt + 1) % unsamp_factor
        else:
            rinds = np.random.randint(sampled_phaenc_len,
                                      size=(sampled_phaenc_len
                                            // unsamp_factor))

        rand_unsamp_lines = sampled_phaenc_ind[rinds]

        tmp = kdat4[l, rand_unsamp_lines, ...]
        output[l, rand_unsamp_lines, ...] = tmp

    return np.reshape(output, kdat.shape)

# %%
def split_shots(kdat: np.ndarray, 
                pe_axis: int = -2, 
                shots: int = 2, 
                pe_reverse: bool = False):
    """split shots within one diffusion encoding
    Input:
        kdat: k-space data, shape [..., ky, ...]
        pe_axis: phase-encoding axis [default: -2]
        shots: number of shots [default: 2]
        pe_reverse: whether the phase-encoding direction is reversed every second shot [default: False]
    
    Output:
        output: shot-split k-space data, shape [shots, ..., ky, ...]
    """
    # find valid phase-encoding lines
    kdat1 = np.swapaxes(kdat, pe_axis, 0)
    kdat2 = np.reshape(kdat1, (kdat1.shape[0], -1))

    kdat3 = np.sum(kdat2, axis=1)
    sampled_phaenc_ind = np.array(np.nonzero(kdat3)).ravel()
    sampled_phaenc_len = len(sampled_phaenc_ind)

    out_shape = [shots] + list(kdat2.shape)
    output = np.zeros_like(kdat2, shape=out_shape)

    phaenc_len = kdat2.shape[0]
    
    if sampled_phaenc_len % 2 == 1: # odd lines
        offset = -1
    else:
        offset = 0

    for l in range(sampled_phaenc_len):
        s = l % shots

        ind_o = sampled_phaenc_ind[l]
        ind_i = sampled_phaenc_ind[l]
        
        if (s % 2 == 1) and (pe_reverse is True):
            ind_i = sampled_phaenc_ind[-l+offset]
            ind_o = phaenc_len - sampled_phaenc_ind[l+offset]

        print('> l %3d i %3d o %3d'%(l, ind_i, ind_o))

        output[s, ind_o, :] = kdat2[ind_i, :]

    output = np.reshape(output, [shots] + list(kdat1.shape))
    output = np.swapaxes(output, 1, pe_axis)

    return output

# %%
def find_nonzero_ky_lines(kdat: np.ndarray, ky_axis: int = -2):
    """find non-zero ky lines in k-space data
    
    Input:
        kdat: k-space data, shape [..., ky, ...]
        ky_axis: phase-encoding axis [default: -2]
    
    Output:
        nonzero_ky_ind: indices of non-zero ky lines, shape [..., N_nonzero_ky]
    """
    assert kdat.shape[-3] == 1  # slice dim must be 1

    kdat_high_shape = kdat.shape[:DIM_ECHO]
    kdat_high_len = np.prod(kdat_high_shape)

    kdat5 = np.reshape(kdat, [-1] + list(kdat.shape[DIM_ECHO:]))

    nonzero_ky_ind = []
    nonzero_ky_len = []

    for h in range(kdat_high_len):

        kdat1 = kdat5[h, ...]        
        kdat1 = np.sum(kdat1, axis=(DIM_ECHO, DIM_COIL, DIM_Z), keepdims=True)

        kdat1 = np.swapaxes(kdat1, ky_axis, 0)
        kdat2 = np.reshape(kdat1, (kdat1.shape[0], -1))

        kdat3 = np.sum(kdat2, axis=1)

        h_ind = np.array(np.nonzero(kdat3)).ravel()
        nonzero_ky_ind.append(h_ind)
        nonzero_ky_len.append(len(h_ind))

    nonzero_ky_len = min(np.array(nonzero_ky_len))
    # print('--> length of nonzero_ky_len: ', nonzero_ky_len)

    for h in range(kdat_high_len):
        nonzero_ky_ind[h] = nonzero_ky_ind[h][:nonzero_ky_len]

    nonzero_ky_ind = np.array(nonzero_ky_ind)
    nonzero_ky_ind = np.reshape(nonzero_ky_ind, list(kdat.shape[:DIM_ECHO]) + [-1])

    return nonzero_ky_ind