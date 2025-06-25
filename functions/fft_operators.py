import math
from functools import partial

import numpy as np
from scipy.interpolate import interp2d

import paddle


class A_functions_fft:
    """
    All input vectors are of shape (Batch, ...).
    All output vectors are of shape (Batch, DataDimension) ---- to remain compatible with code
    """

    def singulars(self):
        """
        Returns a vector containing the singular values. The shape of the vector should be the same as the smaller dimension (like U)
        """
        raise NotImplementedError()

    def add_zeros(self, vec):
        """
        Adds trailing zeros to turn a vector from the small dimension (U) to the big dimension (V)
        """
        raise NotImplementedError()

    def A(self, vec):
        """
        Multiplies the input vector by A
        """
        raise NotImplementedError()

    def At(self, vec):
        """
        Multiplies the input vector by A transposed
        """
        raise NotImplementedError()

    def A_pinv_add_eta(self, vec, eta_reg=0.0001):
        """
        Multiplies the input vector by the pseudo inverse of A
        """
        raise NotImplementedError()

    def invAAt(self, vec):
        """
        Multiplies the input vector by the inverse of AAt
        """
        raise NotImplementedError()


class Deblurring_fft(A_functions_fft):
    def __init__(self, kernel, channels, img_dim, device, eta_reg=0.0001):
        self.img_dim = img_dim
        self.channels = channels
        if kernel.dim() == 1:
            kernel2D = (
                paddle.matmul(x=kernel[:, None], y=kernel[None, :])
                / paddle.sum(x=kernel) ** 2
            )
            self.kernel = kernel2D
        else:
            self.kernel = kernel
        self.eta_reg = eta_reg

    def A(self, vec):
        I = vec.reshape(tuple(vec.shape)[0], self.channels, self.img_dim, self.img_dim)
        out = paddle.zeros(
            shape=[
                tuple(I.shape)[0],
                tuple(I.shape)[1],
                tuple(I.shape)[2],
                tuple(I.shape)[3],
            ]
        ).to(vec.place)
        for ch in range(tuple(out.shape)[1]):
            out[:, ch : ch + 1, :, :] = cconv2_by_fft2(
                I[:, ch : ch + 1, :, :], self.kernel, vec.place, flag_invertB=0
            )
        return out.reshape(tuple(vec.shape)[0], -1)

    def At(self, vec):
        I = vec.reshape(tuple(vec.shape)[0], self.channels, self.img_dim, self.img_dim)
        out = paddle.zeros(
            shape=[
                tuple(I.shape)[0],
                tuple(I.shape)[1],
                tuple(I.shape)[2],
                tuple(I.shape)[3],
            ]
        ).to(vec.place)
        flipped_kernel = paddle.flip(
            x=paddle.flip(x=paddle.conj(x=self.kernel), axis=0), axis=1
        )
        for ch in range(tuple(out.shape)[1]):
            out[:, ch : ch + 1, :, :] = cconv2_by_fft2(
                I[:, ch : ch + 1, :, :], flipped_kernel, vec.place, flag_invertB=0
            )
        return out.reshape(tuple(vec.shape)[0], -1)

    def A_pinv_add_eta(self, vec, eta_reg=None):
        I = vec.reshape(tuple(vec.shape)[0], self.channels, self.img_dim, self.img_dim)
        out = paddle.zeros(
            shape=[
                tuple(I.shape)[0],
                tuple(I.shape)[1],
                tuple(I.shape)[2],
                tuple(I.shape)[3],
            ]
        ).to(vec.place)
        if eta_reg is None:
            eta_reg = self.eta_reg
        flipped_kernel = paddle.flip(
            x=paddle.flip(x=paddle.conj(x=self.kernel), axis=0), axis=1
        )
        for ch in range(tuple(out.shape)[1]):
            temp = cconv2_invAAt_by_fft2(
                I[:, ch : ch + 1, :, :], self.kernel, vec.place, eta=eta_reg
            )
            out[:, ch : ch + 1, :, :] = cconv2_by_fft2(
                temp, flipped_kernel, vec.place, flag_invertB=0
            )
        return out.reshape(tuple(vec.shape)[0], -1)

    def invAAt(self, vec, eta_reg=None):
        I = vec.reshape(tuple(vec.shape)[0], self.channels, self.img_dim, self.img_dim)
        out = paddle.zeros(
            shape=[
                tuple(I.shape)[0],
                tuple(I.shape)[1],
                tuple(I.shape)[2],
                tuple(I.shape)[3],
            ]
        ).to(vec.place)
        if eta_reg is None:
            eta_reg = self.eta_reg
        for ch in range(tuple(out.shape)[1]):
            out[:, ch : ch + 1, :, :] = cconv2_invAAt_by_fft2(
                I[:, ch : ch + 1, :, :], self.kernel, vec.place, eta=eta_reg
            )
        return out.reshape(tuple(vec.shape)[0], -1)

    def AtA_add_eta_inv(self, vec, eta_reg=None):
        I = vec.reshape(tuple(vec.shape)[0], self.channels, self.img_dim, self.img_dim)
        out = paddle.zeros(
            shape=[
                tuple(I.shape)[0],
                tuple(I.shape)[1],
                tuple(I.shape)[2],
                tuple(I.shape)[3],
            ]
        ).to(vec.place)
        if eta_reg is None:
            eta_reg = self.eta_reg
        for ch in range(tuple(out.shape)[1]):
            out[:, ch : ch + 1, :, :] = cconv2_invAAt_by_fft2(
                I[:, ch : ch + 1, :, :], self.kernel, vec.place, eta=eta_reg
            )
        return out.reshape(tuple(vec.shape)[0], -1)


class Superres_fft(A_functions_fft):
    def __init__(self, kernel, channels, img_dim, device, stride=1, eta_reg=0.0001):
        self.img_dim = img_dim
        self.channels = channels
        if kernel.dim() == 1:
            kernel2D = (
                paddle.matmul(x=kernel[:, None], y=kernel[None, :])
                / paddle.sum(x=kernel) ** 2
            )
            self.kernel = kernel2D
        else:
            self.kernel = kernel
        self.ratio = stride
        small_dim = img_dim // stride
        self.small_dim = small_dim
        self.eta_reg = eta_reg

    def A(self, vec):
        I = vec.reshape(tuple(vec.shape)[0], self.channels, self.img_dim, self.img_dim)
        temp = paddle.zeros(
            shape=[tuple(vec.shape)[0], self.channels, self.img_dim, self.img_dim]
        ).to(vec.place)
        for ch in range(self.channels):
            temp[:, ch : ch + 1, :, :] = cconv2_by_fft2(
                I[:, ch : ch + 1, :, :], self.kernel, vec.place, flag_invertB=0
            )
        out = downsample(temp, self.ratio)
        return out.reshape(tuple(vec.shape)[0], -1)

    def At(self, vec):
        I = vec.reshape(
            tuple(vec.shape)[0], self.channels, self.small_dim, self.small_dim
        )
        out = paddle.zeros(
            shape=[tuple(I.shape)[0], self.channels, self.img_dim, self.img_dim]
        ).to(vec.place)
        flipped_kernel = paddle.flip(
            x=paddle.flip(x=paddle.conj(x=self.kernel), axis=0), axis=1
        )
        temp = upsample_MN(I, self.ratio, self.img_dim, self.img_dim)
        for ch in range(self.channels):
            out[:, ch : ch + 1, :, :] = cconv2_by_fft2(
                temp[:, ch : ch + 1, :, :], flipped_kernel, vec.place, flag_invertB=0
            )
        return out.reshape(tuple(vec.shape)[0], -1)

    def invAAt(self, vec, eta_reg=None):
        I = vec.reshape(
            tuple(vec.shape)[0], self.channels, self.small_dim, self.small_dim
        )
        out = paddle.zeros(
            shape=[
                tuple(I.shape)[0],
                tuple(I.shape)[1],
                tuple(I.shape)[2],
                tuple(I.shape)[3],
            ]
        ).to(vec.place)
        if eta_reg is None:
            eta_reg = self.eta_reg
        mk, nk = tuple(self.kernel.shape)[:2]
        bigK = paddle.zeros(shape=(self.img_dim, self.img_dim)).to(vec.place)
        bigK[:mk, :nk] = self.kernel
        bigK = paddle.roll(
            x=bigK, shifts=(-int((mk - 1) / 2), -int((mk - 1) / 2)), axis=(0, 1)
        )
        fft2_K = paddle.fft.fft2(x=bigK)
        h0_full = paddle.real(x=paddle.fft.ifft2(x=paddle.abs(x=fft2_K) ** 2))
        h0 = h0_full[:: self.ratio, :: self.ratio]
        fft2_h0 = paddle.fft.fft2(x=h0)
        inv_fft2_h0 = 1 / (fft2_h0 + eta_reg)
        for ch in range(self.channels):
            out[:, ch : ch + 1, :, :] = paddle.real(
                x=paddle.fft.ifft2(
                    x=paddle.fft.fft2(x=I[:, ch : ch + 1, :, :]) * inv_fft2_h0
                )
            )
        return out.reshape(tuple(vec.shape)[0], -1)

    def A_pinv_add_eta(self, vec, eta_reg=None):
        I = vec.reshape(
            tuple(vec.shape)[0], self.channels, self.small_dim, self.small_dim
        )
        out = paddle.zeros(
            shape=[tuple(I.shape)[0], self.channels, self.img_dim, self.img_dim]
        ).to(vec.place)
        if eta_reg is None:
            eta_reg = self.eta_reg
        mk, nk = tuple(self.kernel.shape)[:2]
        bigK = paddle.zeros(shape=(self.img_dim, self.img_dim)).to(vec.place)
        bigK[:mk, :nk] = self.kernel
        bigK = paddle.roll(
            x=bigK, shifts=(-int((mk - 1) / 2), -int((mk - 1) / 2)), axis=(0, 1)
        )
        fft2_K = paddle.fft.fft2(x=bigK)
        h0_full = paddle.real(x=paddle.fft.ifft2(x=paddle.abs(x=fft2_K) ** 2))
        h0 = h0_full[:: self.ratio, :: self.ratio]
        fft2_h0 = paddle.fft.fft2(x=h0)
        inv_fft2_h0 = 1 / (fft2_h0 + eta_reg)
        temp = paddle.zeros(
            shape=[
                tuple(I.shape)[0],
                tuple(I.shape)[1],
                tuple(I.shape)[2],
                tuple(I.shape)[3],
            ]
        ).to(vec.place)
        for ch in range(self.channels):
            temp[:, ch : ch + 1, :, :] = paddle.real(
                x=paddle.fft.ifft2(
                    x=paddle.fft.fft2(x=I[:, ch : ch + 1, :, :]) * inv_fft2_h0
                )
            )
        flipped_kernel = paddle.flip(
            x=paddle.flip(x=paddle.conj(x=self.kernel), axis=0), axis=1
        )
        temp2 = upsample_MN(temp, self.ratio, self.img_dim, self.img_dim)
        for ch in range(self.channels):
            out[:, ch : ch + 1, :, :] = cconv2_by_fft2(
                temp2[:, ch : ch + 1, :, :], flipped_kernel, vec.place, flag_invertB=0
            )
        return out.reshape(tuple(vec.shape)[0], -1)

    def AtA_add_eta_inv(self, vec, eta_reg=None):
        assert 1, "TODO"


def cconv2_by_fft2(A, B, device, flag_invertB=0, eta=0.01):
    m, n = tuple(A.shape)[2:]
    mb, nb = tuple(B.shape)[:2]
    bigB = paddle.zeros(shape=(m, n)).to(device)
    bigB[:mb, :nb] = B
    bigB = paddle.roll(
        x=bigB, shifts=(-int((mb - 1) / 2), -int((mb - 1) / 2)), axis=(0, 1)
    )
    fft2B = paddle.fft.fft2(x=bigB)
    if flag_invertB:
        fft2B = paddle.conj(x=fft2B) / (paddle.abs(x=fft2B) ** 2 + eta)
    result = paddle.real(x=paddle.fft.ifft2(x=paddle.fft.fft2(x=A) * fft2B))
    return result


def cconv2_invAAt_by_fft2(A, B, device, eta=0.01):
    m, n = tuple(A.shape)[2:]
    mb, nb = tuple(B.shape)[:2]
    bigB = paddle.zeros(shape=(m, n)).to(device)
    bigB[:mb, :nb] = B
    bigB = paddle.roll(
        x=bigB, shifts=(-int((mb - 1) / 2), -int((mb - 1) / 2)), axis=(0, 1)
    )
    fft2B = paddle.fft.fft2(x=bigB)
    fft2B_norm2 = paddle.abs(x=fft2B) ** 2
    inv_fft2B_norm = 1 / (fft2B_norm2 + eta)
    result = paddle.real(x=paddle.fft.ifft2(x=paddle.fft.fft2(x=A) * inv_fft2B_norm))
    return result


def upsample(x, sf=3):
    """s-fold upsampler
    Upsampling the spatial size by filling the new entries with zeros
    x: tensor image, NxCxWxH
    """
    st = 0
    z = paddle.zeros(
        shape=(
            tuple(x.shape)[0],
            tuple(x.shape)[1],
            tuple(x.shape)[2] * sf,
            tuple(x.shape)[3] * sf,
        )
    ).astype(dtype=x.dtype)
    paddle.assign(x, output=z[..., st::sf, st::sf])
    return z


def downsample(x, sf=3):
    """s-fold downsampler
    Keeping the upper-left pixel for each distinct sfxsf patch and discarding the others
    x: tensor image, NxCxWxH
    """
    st = 0
    return x[..., st::sf, st::sf]


def upsample_MN(x, sf=3, M=None, N=None):
    z = upsample(x, sf)
    if M is not None and N is not None:
        z = z[:, :, :M, :N]
    return z


def cubic(x):
    absx = np.abs(x)
    absx2 = absx**2
    absx3 = absx**3
    f = (1.5 * absx3 - 2.5 * absx2 + 1) * (absx <= 1) + (
        -0.5 * absx3 + 2.5 * absx2 - 4 * absx + 2
    ) * ((1 < absx) & (absx <= 2))
    return f


def prepare_cubic_filter(scale):
    kernel_width = 4
    kernel_width = kernel_width / scale
    u = 0
    left = np.floor(u - kernel_width / 2)
    P = np.ceil(kernel_width) + 1
    indices = left + np.arange(0, P, 1)
    weights = scale * cubic(scale * (u - indices))
    weights = np.reshape(weights, [1, weights.size])
    return np.matmul(weights.T, weights)


def matlab_style_gauss2D(shape=(7, 7), sigma=1.6):
    """
    2D gaussian mask - should give the same result as MATLAB's
    fspecial('gaussian',[shape],[sigma])
    """
    m, n = [((ss - 1.0) / 2.0) for ss in shape]
    y, x = np.ogrid[-m : m + 1, -n : n + 1]
    h = np.exp(-(x * x + y * y) / (2.0 * sigma * sigma))
    h[h < np.finfo(h.dtype).eps * h.max_func()] = 0
    sumh = h.sum()
    if sumh != 0:
        h /= sumh
    return h


def shift_pixel(x, sf, upper_left=True):
    """shift pixel for super-resolution with different scale factors
    Args:
        x: WxHxC or WxH
        sf: scale factor
        upper_left: shift direction
    """
    h, w = tuple(x.shape)[:2]
    shift = (sf - 1) * 0.5
    xv, yv = np.arange(0, w, 1.0), np.arange(0, h, 1.0)
    if upper_left:
        x1 = xv + shift
        y1 = yv + shift
    else:
        x1 = xv - shift
        y1 = yv - shift
    x1 = np.clip(x1, 0, w - 1)
    y1 = np.clip(y1, 0, h - 1)
    if x.ndim == 2:
        x = interp2d(xv, yv, x)(x1, y1)
    if x.ndim == 3:
        for i in range(tuple(x.shape)[-1]):
            x[:, :, i] = interp2d(xv, yv, x[:, :, i])(x1, y1)
    return x
