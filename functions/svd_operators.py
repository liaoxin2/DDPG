import math

import numpy as np

import paddle


class A_functions:
    """
    A class replacing the SVD of a matrix A, perhaps efficiently.
    All input vectors are of shape (Batch, ...).
    All output vectors are of shape (Batch, DataDimension).
    """

    def V(self, vec):
        """
        Multiplies the input vector by V
        """
        raise NotImplementedError()

    def Vt(self, vec):
        """
        Multiplies the input vector by V transposed
        """
        raise NotImplementedError()

    def U(self, vec):
        """
        Multiplies the input vector by U
        """
        raise NotImplementedError()

    def Ut(self, vec):
        """
        Multiplies the input vector by U transposed
        """
        raise NotImplementedError()

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
        temp = self.Vt(vec)
        singulars = self.singulars()
        return self.U(singulars * temp[:, : tuple(singulars.shape)[0]])

    def At(self, vec):
        """
        Multiplies the input vector by A transposed
        """
        temp = self.Ut(vec)
        singulars = self.singulars()
        return self.V(self.add_zeros(singulars * temp[:, : tuple(singulars.shape)[0]]))

    def A_pinv(self, vec):
        """
        Multiplies the input vector by the pseudo inverse of A
        """
        temp = self.Ut(vec)
        singulars = self.singulars()
        factors = 1.0 / singulars
        factors[singulars == 0] = 0.0
        temp[:, : tuple(singulars.shape)[0]] = (
            temp[:, : tuple(singulars.shape)[0]] * factors
        )
        return self.V(self.add_zeros(temp))

    def A_pinv_add_eta(self, vec, eta=0):
        """
        Multiplies the input vector by the pseudo inverse of A with loading eta
        """
        temp = self.Ut(vec)
        singulars = self.singulars()
        factors = singulars / (singulars * singulars + eta)
        temp[:, : tuple(singulars.shape)[0]] = (
            temp[:, : tuple(singulars.shape)[0]] * factors
        )
        return self.V(self.add_zeros(temp))

    def AtA_add_eta_inv(self, vec, eta=1e-06):
        """
        Multiplies the input vector by the pseudo inverse of A with loading eta
        """
        temp = self.Vt(vec)
        singulars = self.singulars()
        extended_singulars = self.add_zeros(singulars[None, :])
        factors = 1 / (extended_singulars * extended_singulars + eta)
        temp = temp * factors
        return self.V(temp)


class CS(A_functions):
    def __init__(self, channels, img_dim, ratio, device):
        self.img_dim = img_dim
        self.channels = channels
        self.y_dim = img_dim // 32
        self.ratio = 32
        A = paddle.randn(shape=[32**2, 32**2]).to(device)
        tmp_u, tmp_s, tmp_v = paddle.linalg.svd(x=A, full_matrices=not False)
        _, _, self.V_small = tmp_u, tmp_s, tmp_v.conj().t()
        self.Vt_small = self.V_small.transpose([1,0])
        self.singulars_small = paddle.ones(shape=int(32 * 32 * ratio))
        self.cs_size = self.singulars_small.shape[0]

    def V(self, vec):
        temp = vec.clone().reshape(tuple(vec.shape)[0], -1)
        patches = paddle.zeros(
            shape=[vec.shape[0], self.channels * self.y_dim**2, self.ratio**2]
        )
        patches[:, :, : self.cs_size] = (
            temp[:, : self.channels * self.y_dim**2 * self.cs_size]
            .contiguous()
            .reshape(vec.shape[0], -1, self.cs_size)
        )
        patches[:, :, self.cs_size :] = (
            temp[:, self.channels * self.y_dim**2 * self.cs_size :]
            .contiguous()
            .reshape(vec.shape[0], self.channels * self.y_dim**2, -1)
        )
        patches = paddle.matmul(
            x=self.V_small, y=patches.reshape(-1, self.ratio**2, 1)
        ).reshape(tuple(vec.shape)[0], self.channels, -1, self.ratio**2)
        patches_orig = patches.reshape(
            tuple(vec.shape)[0],
            self.channels,
            self.y_dim,
            self.y_dim,
            self.ratio,
            self.ratio,
        )
        recon = patches_orig.transpose(perm=[0, 1, 2, 4, 3, 5]).contiguous()
        recon = recon.reshape(tuple(vec.shape)[0], self.channels * self.img_dim**2)
        return recon

    def Vt(self, vec):
        patches = vec.clone().reshape(
            tuple(vec.shape)[0], self.channels, self.img_dim, self.img_dim
        )
        patches = patches.unfold(axis=2, size=self.ratio, step=self.ratio).unfold(
            axis=3, size=self.ratio, step=self.ratio
        )
        patches = patches.contiguous().reshape(
            tuple(vec.shape)[0], self.channels, -1, self.ratio**2
        )
        patches = paddle.matmul(
            x=self.Vt_small, y=patches.reshape(-1, self.ratio**2, 1)
        ).reshape(tuple(vec.shape)[0], self.channels, -1, self.ratio**2)
        recon = paddle.zeros(
            shape=[tuple(vec.shape)[0], self.channels * self.img_dim**2]
        )
        recon[:, : self.channels * self.y_dim**2 * self.cs_size] = (
            patches[:, :, :, : self.cs_size]
            .contiguous()
            .reshape(tuple(vec.shape)[0], -1)
        )
        recon[:, self.channels * self.y_dim**2 * self.cs_size :] = (
            patches[:, :, :, self.cs_size :]
            .contiguous()
            .reshape(tuple(vec.shape)[0], -1)
        )
        return recon

    def U(self, vec):
        return vec.clone().reshape(tuple(vec.shape)[0], -1)

    def Ut(self, vec):
        return vec.clone().reshape(tuple(vec.shape)[0], -1)

    def singulars(self):
        return self.singulars_small.tile(repeat_times=self.channels * self.y_dim**2)

    def add_zeros(self, vec):
        reshaped = vec.clone().reshape(tuple(vec.shape)[0], -1)
        temp = paddle.zeros(
            shape=(tuple(vec.shape)[0], self.channels * self.img_dim**2)
        )
        temp[:, : tuple(reshaped.shape)[1]] = reshaped
        return temp


def color2gray(x):
    x = x[:, 0:1, :, :] * 0.3333 + x[:, 1:2, :, :] * 0.3334 + x[:, 2:, :, :] * 0.3333
    return x


def gray2color(x):
    base = 0.3333**2 + 0.3334**2 + 0.3333**2
    return paddle.stack(
        x=(x * 0.3333 / base, x * 0.3334 / base, x * 0.3333 / base), axis=1
    )


class GeneralA(A_functions):
    def mat_by_vec(self, M, v):
        vshape = tuple(v.shape)[1]
        if len(tuple(v.shape)) > 2:
            vshape = vshape * tuple(v.shape)[2]
        if len(tuple(v.shape)) > 3:
            vshape = vshape * tuple(v.shape)[3]
        return paddle.matmul(x=M, y=v.view(tuple(v.shape)[0], vshape, 1)).view(
            tuple(v.shape)[0], tuple(M.shape)[0]
        )

    def __init__(self, A):
        tmp_u, tmp_s, tmp_v = paddle.linalg.svd(x=A, full_matrices=not False)
        self._U, self._singulars, self._V = tmp_u, tmp_s, tmp_v.conj().t()
        self._Vt = self._V.transpose([1,0])
        self._Ut = self._U.transpose([1,0])
        ZERO = 0.001
        self._singulars[self._singulars < ZERO] = 0
        print(len([x.item() for x in self._singulars if x == 0]))

    def V(self, vec):
        return self.mat_by_vec(self._V, vec.clone())

    def Vt(self, vec):
        return self.mat_by_vec(self._Vt, vec.clone())

    def U(self, vec):
        return self.mat_by_vec(self._U, vec.clone())

    def Ut(self, vec):
        return self.mat_by_vec(self._Ut, vec.clone())

    def singulars(self):
        return self._singulars

    def add_zeros(self, vec):
        out = paddle.zeros(shape=[tuple(vec.shape)[0], tuple(self._V.shape)[0]])
        out[:, : tuple(self._U.shape)[0]] = vec.clone().reshape(tuple(vec.shape)[0], -1)
        return out


class WalshHadamardCS(A_functions):
    def fwht(self, vec):
        a = vec.reshape(tuple(vec.shape)[0], self.channels, self.img_dim**2)
        h = 1
        while h < self.img_dim**2:
            a = a.reshape(tuple(vec.shape)[0], self.channels, -1, h * 2)
            b = a.clone()
            a[:, :, :, :h] = b[:, :, :, :h] + b[:, :, :, h : 2 * h]
            a[:, :, :, h : 2 * h] = b[:, :, :, :h] - b[:, :, :, h : 2 * h]
            h *= 2
        a = (
            a.reshape(tuple(vec.shape)[0], self.channels, self.img_dim**2)
            / self.img_dim
        )
        return a

    def __init__(self, channels, img_dim, ratio, perm, device):
        self.channels = channels
        self.img_dim = img_dim
        self.ratio = ratio
        self.perm = perm
        self._singulars = paddle.ones(shape=channels * img_dim**2 // ratio)

    def V(self, vec):
        temp = paddle.zeros(
            shape=[tuple(vec.shape)[0], self.channels, self.img_dim**2]
        )
        temp[:, :, self.perm] = (
            vec.clone()
            .reshape(tuple(vec.shape)[0], -1, self.channels)
            .transpose(perm=[0, 2, 1])
        )
        return self.fwht(temp).reshape(tuple(vec.shape)[0], -1)

    def Vt(self, vec):
        return (
            self.fwht(vec.clone())[:, :, self.perm]
            .transpose(perm=[0, 2, 1])
            .reshape(tuple(vec.shape)[0], -1)
        )

    def U(self, vec):
        return vec.clone().reshape(tuple(vec.shape)[0], -1)

    def Ut(self, vec):
        return vec.clone().reshape(tuple(vec.shape)[0], -1)

    def singulars(self):
        return self._singulars

    def add_zeros(self, vec):
        out = paddle.zeros(
            shape=[tuple(vec.shape)[0], self.channels * self.img_dim**2]
        )
        out[:, : self.channels * self.img_dim**2 // self.ratio] = vec.clone().reshape(
            tuple(vec.shape)[0], -1
        )
        return out


class Inpainting(A_functions):
    def __init__(self, channels, img_dim, missing_indices, device):
        self.channels = channels
        self.img_dim = img_dim
        self._singulars = paddle.ones(
            shape=channels * img_dim**2 - tuple(missing_indices.shape)[0]
        ).to(device)
        self.missing_indices = missing_indices
        self.kept_indices = (
            paddle.to_tensor(
                data=[
                    i
                    for i in range(channels * img_dim**2)
                    if i not in missing_indices
                ],
                dtype="float32",
            )
            .to(device)
            .astype(dtype="int64")
        )

    def V(self, vec):
        temp = vec.clone().reshape(tuple(vec.shape)[0], -1)
        out = paddle.zeros_like(x=temp)
        out[:, self.kept_indices] = temp[:, : tuple(self.kept_indices.shape)[0]]
        out[:, self.missing_indices] = temp[:, tuple(self.kept_indices.shape)[0] :]
        return (
            out.reshape(tuple(vec.shape)[0], -1, self.channels)
            .transpose(perm=[0, 2, 1])
            .reshape(tuple(vec.shape)[0], -1)
        )

    def Vt(self, vec):
        temp = (
            vec.clone()
            .reshape(tuple(vec.shape)[0], self.channels, -1)
            .transpose(perm=[0, 2, 1])
            .reshape(tuple(vec.shape)[0], -1)
        )
        out = paddle.zeros_like(x=temp)
        out[:, : tuple(self.kept_indices.shape)[0]] = temp[:, self.kept_indices]
        out[:, tuple(self.kept_indices.shape)[0] :] = temp[:, self.missing_indices]
        return out

    def U(self, vec):
        return vec.clone().reshape(tuple(vec.shape)[0], -1)

    def Ut(self, vec):
        return vec.clone().reshape(tuple(vec.shape)[0], -1)

    def singulars(self):
        return self._singulars

    def add_zeros(self, vec):
        temp = paddle.zeros(
            shape=(tuple(vec.shape)[0], self.channels * self.img_dim**2)
        )
        reshaped = vec.clone().reshape(tuple(vec.shape)[0], -1)
        temp[:, : tuple(reshaped.shape)[1]] = reshaped
        return temp


class Denoising(A_functions):
    def __init__(self, channels, img_dim, device):
        self._singulars = paddle.ones(shape=channels * img_dim**2)

    def V(self, vec):
        return vec.clone().reshape(tuple(vec.shape)[0], -1)

    def Vt(self, vec):
        return vec.clone().reshape(tuple(vec.shape)[0], -1)

    def U(self, vec):
        return vec.clone().reshape(tuple(vec.shape)[0], -1)

    def Ut(self, vec):
        return vec.clone().reshape(tuple(vec.shape)[0], -1)

    def singulars(self):
        return self._singulars

    def add_zeros(self, vec):
        return vec.clone().reshape(tuple(vec.shape)[0], -1)


class SuperResolution(A_functions):
    def __init__(self, channels, img_dim, ratio, device):
        assert img_dim % ratio == 0
        self.img_dim = img_dim
        self.channels = channels
        self.y_dim = img_dim // ratio
        self.ratio = ratio
        A = paddle.to_tensor(data=[[1 / ratio**2] * ratio**2], dtype="float32").to(
            device
        )
        tmp_u, tmp_s, tmp_v = paddle.linalg.svd(x=A, full_matrices=not False)
        self.U_small, self.singulars_small, self.V_small = (
            tmp_u,
            tmp_s,
            tmp_v.conj().t(),
        )
        self.Vt_small = self.V_small.transpose([1,0])

    def V(self, vec):
        temp = vec.clone().reshape(tuple(vec.shape)[0], -1)
        patches = paddle.zeros(
            shape=[tuple(vec.shape)[0], self.channels, self.y_dim**2, self.ratio**2]
        )
        patches[:, :, :, 0] = temp[:, : self.channels * self.y_dim**2].view(
            tuple(vec.shape)[0], self.channels, -1
        )
        for idx in range(self.ratio**2 - 1):
            patches[:, :, :, idx + 1] = temp[
                :, self.channels * self.y_dim**2 + idx :: self.ratio**2 - 1
            ].view(tuple(vec.shape)[0], self.channels, -1)
        patches = paddle.matmul(
            x=self.V_small, y=patches.reshape(-1, self.ratio**2, 1)
        ).reshape(tuple(vec.shape)[0], self.channels, -1, self.ratio**2)
        patches_orig = patches.reshape(
            tuple(vec.shape)[0],
            self.channels,
            self.y_dim,
            self.y_dim,
            self.ratio,
            self.ratio,
        )
        recon = patches_orig.transpose(perm=[0, 1, 2, 4, 3, 5]).contiguous()
        recon = recon.reshape(tuple(vec.shape)[0], self.channels * self.img_dim**2)
        return recon

    def Vt(self, vec):
        patches = vec.clone().reshape(
            tuple(vec.shape)[0], self.channels, self.img_dim, self.img_dim
        )
        patches = patches.unfold(axis=2, size=self.ratio, step=self.ratio).unfold(
            axis=3, size=self.ratio, step=self.ratio
        )
        unfold_shape = tuple(patches.shape)
        patches = patches.contiguous().reshape(
            tuple(vec.shape)[0], self.channels, -1, self.ratio**2
        )
        patches = paddle.matmul(
            x=self.Vt_small, y=patches.reshape(-1, self.ratio**2, 1)
        ).reshape(tuple(vec.shape)[0], self.channels, -1, self.ratio**2)
        recon = paddle.zeros(
            shape=[tuple(vec.shape)[0], self.channels * self.img_dim**2]
        )
        recon[:, : self.channels * self.y_dim**2] = patches[:, :, :, 0].view(
            tuple(vec.shape)[0], self.channels * self.y_dim**2
        )
        for idx in range(self.ratio**2 - 1):
            recon[
                :, self.channels * self.y_dim**2 + idx :: self.ratio**2 - 1
            ] = patches[:, :, :, idx + 1].view(
                tuple(vec.shape)[0], self.channels * self.y_dim**2
            )
        return recon

    def U(self, vec):
        return self.U_small[0, 0] * vec.clone().reshape(tuple(vec.shape)[0], -1)

    def Ut(self, vec):
        return self.U_small[0, 0] * vec.clone().reshape(tuple(vec.shape)[0], -1)

    def singulars(self):
        return self.singulars_small.tile(repeat_times=self.channels * self.y_dim**2)

    def add_zeros(self, vec):
        reshaped = vec.clone().reshape(tuple(vec.shape)[0], -1)
        temp = paddle.zeros(
            shape=(tuple(vec.shape)[0], tuple(reshaped.shape)[1] * self.ratio**2)
        )
        temp[:, : tuple(reshaped.shape)[1]] = reshaped
        return temp


class Colorization(A_functions):
    def __init__(self, img_dim, device):
        self.channels = 3
        self.img_dim = img_dim
        A = paddle.to_tensor(data=[[0.3333, 0.3334, 0.3333]], dtype="float32").to(
            device
        )
        tmp_u, tmp_s, tmp_v = paddle.linalg.svd(x=A, full_matrices=not False)
        self.U_small, self.singulars_small, self.V_small = (
            tmp_u,
            tmp_s,
            tmp_v.conj().t(),
        )
        self.Vt_small = self.V_small.transpose([1,0])

    def V(self, vec):
        needles = (
            vec.clone()
            .reshape(tuple(vec.shape)[0], self.channels, -1)
            .transpose(perm=[0, 2, 1])
        )
        needles = paddle.matmul(
            x=self.V_small, y=needles.reshape(-1, self.channels, 1)
        ).reshape(tuple(vec.shape)[0], -1, self.channels)
        recon = needles.transpose(perm=[0, 2, 1])
        return recon.reshape(tuple(vec.shape)[0], -1)

    def Vt(self, vec):
        needles = (
            vec.clone()
            .reshape(tuple(vec.shape)[0], self.channels, -1)
            .transpose(perm=[0, 2, 1])
        )
        needles = paddle.matmul(
            x=self.Vt_small, y=needles.reshape(-1, self.channels, 1)
        ).reshape(tuple(vec.shape)[0], -1, self.channels)
        recon = needles.transpose(perm=[0, 2, 1]).reshape(tuple(vec.shape)[0], -1)
        return recon

    def U(self, vec):
        return self.U_small[0, 0] * vec.clone().reshape(tuple(vec.shape)[0], -1)

    def Ut(self, vec):
        return self.U_small[0, 0] * vec.clone().reshape(tuple(vec.shape)[0], -1)

    def singulars(self):
        return self.singulars_small.tile(repeat_times=self.img_dim**2)

    def add_zeros(self, vec):
        reshaped = vec.clone().reshape(tuple(vec.shape)[0], -1)
        temp = paddle.zeros(
            shape=(tuple(vec.shape)[0], self.channels * self.img_dim**2)
        )
        temp[:, : self.img_dim**2] = reshaped
        return temp


class WalshAadamardCS(A_functions):
    def fwht(self, vec):
        a = vec.reshape(tuple(vec.shape)[0], self.channels, self.img_dim**2)
        h = 1
        while h < self.img_dim**2:
            a = a.reshape(tuple(vec.shape)[0], self.channels, -1, h * 2)
            b = a.clone()
            a[:, :, :, :h] = b[:, :, :, :h] + b[:, :, :, h : 2 * h]
            a[:, :, :, h : 2 * h] = b[:, :, :, :h] - b[:, :, :, h : 2 * h]
            h *= 2
        a = (
            a.reshape(tuple(vec.shape)[0], self.channels, self.img_dim**2)
            / self.img_dim
        )
        return a

    def __init__(self, channels, img_dim, ratio, perm, device):
        self.channels = channels
        self.img_dim = img_dim
        self.ratio = ratio
        self.perm = perm
        self._singulars = paddle.ones(shape=channels * img_dim**2 // ratio)

    def V(self, vec):
        temp = paddle.zeros(
            shape=[tuple(vec.shape)[0], self.channels, self.img_dim**2]
        )
        temp[:, :, self.perm] = (
            vec.clone()
            .reshape(tuple(vec.shape)[0], -1, self.channels)
            .transpose(perm=[0, 2, 1])
        )
        return self.fwht(temp).reshape(tuple(vec.shape)[0], -1)

    def Vt(self, vec):
        return (
            self.fwht(vec.clone())[:, :, self.perm]
            .transpose(perm=[0, 2, 1])
            .reshape(tuple(vec.shape)[0], -1)
        )

    def U(self, vec):
        return vec.clone().reshape(tuple(vec.shape)[0], -1)

    def Ut(self, vec):
        return vec.clone().reshape(tuple(vec.shape)[0], -1)

    def singulars(self):
        return self._singulars

    def add_zeros(self, vec):
        out = paddle.zeros(
            shape=[tuple(vec.shape)[0], self.channels * self.img_dim**2]
        )
        out[:, : self.channels * self.img_dim**2 // self.ratio] = vec.clone().reshape(
            tuple(vec.shape)[0], -1
        )
        return out


class SRConv(A_functions):
    def mat_by_img(self, M, v, dim):
        return paddle.matmul(
            x=M, y=v.reshape(tuple(v.shape)[0] * self.channels, dim, dim)
        ).reshape(tuple(v.shape)[0], self.channels, tuple(M.shape)[0], dim)

    def img_by_mat(self, v, M, dim):
        return paddle.matmul(
            x=v.reshape(tuple(v.shape)[0] * self.channels, dim, dim), y=M
        ).reshape(tuple(v.shape)[0], self.channels, dim, tuple(M.shape)[1])

    def __init__(self, kernel, channels, img_dim, device, stride=1):
        self.img_dim = img_dim
        self.channels = channels
        self.ratio = stride
        small_dim = img_dim // stride
        self.small_dim = small_dim
        A_small = paddle.zeros(shape=[small_dim, img_dim])
        for i in range(stride // 2, img_dim + stride // 2, stride):
            for j in range(
                i - tuple(kernel.shape)[0] // 2, i + tuple(kernel.shape)[0] // 2
            ):
                j_effective = j
                if j_effective < 0:
                    j_effective = -j_effective - 1
                if j_effective >= img_dim:
                    j_effective = img_dim - 1 - (j_effective - img_dim)
                A_small[i // stride, j_effective] += kernel[
                    j - i + tuple(kernel.shape)[0] // 2
                ]
        tmp_u, tmp_s, tmp_v = paddle.linalg.svd(x=A_small, full_matrices=not False)
        self.U_small, self.singulars_small, self.V_small = (
            tmp_u,
            tmp_s,
            tmp_v.conj().t(),
        )
        ZERO = 0.03
        self.singulars_small[self.singulars_small < ZERO] = 0
        self._singulars = paddle.matmul(
            x=self.singulars_small.reshape(small_dim, 1),
            y=self.singulars_small.reshape(1, small_dim),
        ).reshape(small_dim**2)
        self._perm = (
            paddle.to_tensor(
                data=[
                    (self.img_dim * i + j)
                    for i in range(self.small_dim)
                    for j in range(self.small_dim)
                ]
                + [
                    (self.img_dim * i + j)
                    for i in range(self.small_dim)
                    for j in range(self.small_dim, self.img_dim)
                ],
                dtype="float32",
            )
            .to(device)
            .astype(dtype="int64")
        )

    def V(self, vec):
        temp = paddle.zeros(
            shape=[tuple(vec.shape)[0], self.img_dim**2, self.channels]
        )
        temp[:, self._perm, :] = vec.clone().reshape(
            tuple(vec.shape)[0], self.img_dim**2, self.channels
        )[:, : tuple(self._perm.shape)[0], :]
        temp[:, tuple(self._perm.shape)[0] :, :] = vec.clone().reshape(
            tuple(vec.shape)[0], self.img_dim**2, self.channels
        )[:, tuple(self._perm.shape)[0] :, :]
        temp = temp.transpose(perm=[0, 2, 1])
        out = self.mat_by_img(self.V_small, temp, self.img_dim)
        out = self.img_by_mat(
            out,
            self.V_small.transpose([1,0]),
            self.img_dim,
        ).reshape(tuple(vec.shape)[0], -1)
        return out

    def Vt(self, vec):
        temp = self.mat_by_img(
            self.V_small.transpose([1,0]),
            vec.clone(),
            self.img_dim,
        )
        temp = self.img_by_mat(temp, self.V_small, self.img_dim).reshape(
            tuple(vec.shape)[0], self.channels, -1
        )
        temp[:, :, : tuple(self._perm.shape)[0]] = temp[:, :, self._perm]
        temp = temp.transpose(perm=[0, 2, 1])
        return temp.reshape(tuple(vec.shape)[0], -1)

    def U(self, vec):
        temp = paddle.zeros(
            shape=[tuple(vec.shape)[0], self.small_dim**2, self.channels]
        )
        temp[:, : self.small_dim**2, :] = vec.clone().reshape(
            tuple(vec.shape)[0], self.small_dim**2, self.channels
        )
        temp = temp.transpose(perm=[0, 2, 1])
        out = self.mat_by_img(self.U_small, temp, self.small_dim)
        out = self.img_by_mat(
            out,
            self.U_small.transpose([1,0]),
            self.small_dim,
        ).reshape(tuple(vec.shape)[0], -1)
        return out

    def Ut(self, vec):
        temp = self.mat_by_img(
            self.U_small.transpose([1,0]),
            vec.clone(),
            self.small_dim,
        )
        temp = self.img_by_mat(temp, self.U_small, self.small_dim).reshape(
            tuple(vec.shape)[0], self.channels, -1
        )
        temp = temp.transpose(perm=[0, 2, 1])
        return temp.reshape(tuple(vec.shape)[0], -1)

    def singulars(self):
        return self._singulars.repeat_interleave(repeats=3).reshape(-1)

    def add_zeros(self, vec):
        reshaped = vec.clone().reshape(tuple(vec.shape)[0], -1)
        temp = paddle.zeros(
            shape=(tuple(vec.shape)[0], tuple(reshaped.shape)[1] * self.ratio**2)
        )
        temp[:, : tuple(reshaped.shape)[1]] = reshaped
        return temp


class SRConv_NonSquare(A_functions):
    def mat_by_img(self, M, v, height, width):
        # Reshape the input tensor to flatten spatial dimensions
        v_reshaped = v.reshape([v.shape[0]*self.channels, height, width]) 
        result = paddle.matmul(M, v_reshaped)
        return result.reshape([v.shape[0], self.channels, M.shape[0], width])

    def img_by_mat(self, v, M, height, width):
        # Reshape tensor to flatten spatial dimensions
        v_flat = v.reshape([v.shape[0]*self.channels, height, width])
        result = paddle.matmul(v_flat, M)
        return result.reshape([v.shape[0], self.channels, height, M.shape[1]])
        
    def __init__(self, kernel, channels, img_height, img_width, device, stride=1):
        self.img_height = img_height
        self.img_width = img_width
        self.channels = channels
        self.ratio = stride
        small_height = img_height // stride
        small_width = img_width // stride
        self.small_height = small_height
        self.small_width = small_width

        # Build A_small_h
        A_small_h = paddle.zeros([small_height, img_height], dtype='float32')
        for i in range(stride // 2, img_height + stride // 2, stride):
            if i // stride >= small_height:
                break
            for j in range(i - kernel.shape[0] // 2, i + kernel.shape[0] // 2):
                j_effective = j
                # Reflective padding
                if j_effective < 0:
                    j_effective = -j_effective - 1
                elif j_effective >= img_height:
                    j_effective = (img_height - 1) - (j_effective - img_height)
                # Calculate the index for kernel
                kernel_idx = j - i + kernel.shape[0] // 2
                A_small_h[i // stride, j_effective] += kernel[kernel_idx]

        # Build A_small_w
        A_small_w = paddle.zeros([small_width, img_width], dtype='float32')
        for i in range(stride // 2, img_width + stride // 2, stride):
            if i // stride >= small_width:
                break
            for j in range(i - kernel.shape[0] // 2, i + kernel.shape[0] // 2):
                j_effective = j
                if j_effective < 0:
                    j_effective = -j_effective - 1
                elif j_effective >= img_width:
                    j_effective = (img_width - 1) - (j_effective - img_width)
                kernel_idx = j - i + kernel.shape[0] // 2
                A_small_w[i // stride, j_effective] += kernel[kernel_idx]

        # SVD
        self.U_small_h, self.singulars_small_h, self.V_small_h = paddle.linalg.svd(A_small_h, full_matrices=True)
        self.U_small_w, self.singulars_small_w, self.V_small_w = paddle.linalg.svd(A_small_w, full_matrices=True)

        # Zero out small singular values
        ZERO = 3e-2
        self.singulars_small_h[self.singulars_small_h < ZERO] = 0
        self.singulars_small_w[self.singulars_small_w < ZERO] = 0

        # Compute the product of singular values for the big matrix
        self._singulars = (self.singulars_small_h.reshape([small_height, 1]) @ self.singulars_small_w.reshape([1, small_width])).reshape([-1])

        # Generate permutation indices to match the singular values
        self._perm = paddle.to_tensor([
            self.img_width * i + j for i in range(self.small_height) for j in range(self.small_width)
        ] + [
            self.img_width * i + j for i in range(self.small_height) for j in range(self.small_width, self.img_width)
        ], dtype='int64')

    def V(self, vec):
        # Inverse permutation
        temp = paddle.zeros([vec.shape[0], self.img_height * self.img_width, self.channels], dtype=vec.dtype)
        temp[:, self._perm, :] = vec.reshape([vec.shape[0], self.img_height * self.img_width, self.channels])[:, :self._perm.shape[0], :]
        temp[:, self._perm.shape[0]:, :] = vec.reshape([vec.shape[0], self.img_height * self.img_width, self.channels])[:, self._perm.shape[0]:, :]
        temp = temp.transpose([0, 2, 1])
        # Multiply with V matrices
        out = self.mat_by_img(self.V_small_h.transpose([1, 0]), temp, self.img_height, self.img_width)
        out = self.img_by_mat(out, self.V_small_w, self.img_height, self.img_width).reshape([vec.shape[0], -1])
        return out

    def Vt(self, vec):
        # Multiply with V transpose matrices
        temp = self.mat_by_img(self.V_small_h, vec.clone(), self.img_height, self.img_width)
        temp = self.img_by_mat(temp, self.V_small_w.transpose([1, 0]), self.img_height, self.img_width).reshape([vec.shape[0], self.channels, -1])

        # Inverse permutation
        temp[:, :, :self._perm.shape[0]] = temp[:, :, self._perm]
        temp = temp.transpose([0, 2, 1])
        return temp.reshape([vec.shape[0], -1])

    def U(self, vec):
        # Inverse permutation
        temp = paddle.zeros([vec.shape[0], self.small_height * self.small_width, self.channels], dtype=vec.dtype)
        temp[:, :self.small_height * self.small_width, :] = vec.reshape([vec.shape[0], self.small_height * self.small_width, self.channels])
        temp = temp.transpose([0, 2, 1])
        out = self.mat_by_img(self.U_small_h, temp, self.small_height, self.small_width)
        out = self.img_by_mat(out, self.U_small_w.transpose([1, 0]), self.small_height, self.small_width).reshape([vec.shape[0], -1])
        return out

    def Ut(self, vec):
        # Multiply with U transpose matrices
        temp = self.mat_by_img(self.U_small_h.transpose([1, 0]), vec.clone(), self.small_height, self.small_width)
        temp = self.img_by_mat(temp, self.U_small_w, self.small_height, self.small_width).reshape([vec.shape[0], self.channels, -1])
        # Inverse permutation
        temp = temp.transpose([0, 2, 1])
        return temp.reshape([vec.shape[0], -1])

    def singulars(self):
        return self._singulars.repeat_interleave(self.channels)

    def add_zeros(self, vec):
        reshaped = vec.reshape([vec.shape[0], -1])
        temp = paddle.zeros([vec.shape[0], self.img_height * self.img_width], dtype=vec.dtype)
        temp[:, :reshaped.shape[1]] = reshaped
        return temp


class Deblurring(A_functions):
    def mat_by_img(self, M, v):
        return paddle.matmul(
            x=M,
            y=v.reshape(tuple(v.shape)[0] * self.channels, self.img_dim, self.img_dim),
        ).reshape(tuple(v.shape)[0], self.channels, tuple(M.shape)[0], self.img_dim)

    def img_by_mat(self, v, M):
        return paddle.matmul(
            x=v.reshape(tuple(v.shape)[0] * self.channels, self.img_dim, self.img_dim),
            y=M,
        ).reshape(tuple(v.shape)[0], self.channels, self.img_dim, tuple(M.shape)[1])

    def __init__(self, kernel, channels, img_dim, device, ZERO=0.03):
        self.img_dim = img_dim
        self.channels = channels
        A_small = paddle.zeros(shape=[img_dim, img_dim])
        for i in range(img_dim):
            for j in range(
                i - math.floor(tuple(kernel.shape)[0] / 2),
                i + math.ceil(tuple(kernel.shape)[0] / 2),
            ):
                if j < 0 or j >= img_dim:
                    continue
                A_small[i, j] = kernel[j - i + tuple(kernel.shape)[0] // 2]
        tmp_u, tmp_s, tmp_v = paddle.linalg.svd(x=A_small, full_matrices=not False)
        self.U_small, self.singulars_small, self.V_small = (
            tmp_u,
            tmp_s,
            tmp_v.conj().t(),
        )
        self.singulars_small_orig = self.singulars_small.clone()
        self.singulars_small[self.singulars_small < ZERO] = 0
        self._singulars_orig = paddle.matmul(
            x=self.singulars_small_orig.reshape(img_dim, 1),
            y=self.singulars_small_orig.reshape(1, img_dim),
        ).reshape(img_dim**2)
        self._singulars = paddle.matmul(
            x=self.singulars_small.reshape(img_dim, 1),
            y=self.singulars_small.reshape(1, img_dim),
        ).reshape(img_dim**2)
        self._singulars, self._perm = paddle.sort(
            descending=True, x=self._singulars
        ), paddle.argsort(descending=True, x=self._singulars)
        self._singulars_orig = self._singulars_orig[self._perm]

    def V(self, vec):
        temp = paddle.zeros(
            shape=[tuple(vec.shape)[0], self.img_dim**2, self.channels]
        )
        temp[:, self._perm, :] = vec.clone().reshape(
            tuple(vec.shape)[0], self.img_dim**2, self.channels
        )
        temp = temp.transpose(perm=[0, 2, 1])
        out = self.mat_by_img(self.V_small, temp)
        out = self.img_by_mat(
            out, self.V_small.transpose([1,0])
        ).reshape(tuple(vec.shape)[0], -1)
        return out

    def Vt(self, vec):
        temp = self.mat_by_img(
            self.V_small.transpose([1,0]), vec.clone()
        )
        temp = self.img_by_mat(temp, self.V_small).reshape(
            tuple(vec.shape)[0], self.channels, -1
        )
        temp = temp[:, :, self._perm].transpose(perm=[0, 2, 1])
        return temp.reshape(tuple(vec.shape)[0], -1)

    def U(self, vec):
        temp = paddle.zeros(
            shape=[tuple(vec.shape)[0], self.img_dim**2, self.channels]
        )
        temp[:, self._perm, :] = vec.clone().reshape(
            tuple(vec.shape)[0], self.img_dim**2, self.channels
        )
        temp = temp.transpose(perm=[0, 2, 1])
        out = self.mat_by_img(self.U_small, temp)
        out = self.img_by_mat(
            out, self.U_small.transpose([1,0])
        ).reshape(tuple(vec.shape)[0], -1)
        return out

    def Ut(self, vec):
        temp = self.mat_by_img(
            self.U_small.transpose([1,0]), vec.clone()
        )
        temp = self.img_by_mat(temp, self.U_small).reshape(
            tuple(vec.shape)[0], self.channels, -1
        )
        temp = temp[:, :, self._perm].transpose(perm=[0, 2, 1])
        return temp.reshape(tuple(vec.shape)[0], -1)

    def singulars(self):
        return self._singulars.tile(repeat_times=[1, 3]).reshape(-1)

    def add_zeros(self, vec):
        return vec.clone().reshape(tuple(vec.shape)[0], -1)

    def A_pinv(self, vec, eps_reg=0):
        temp = self.Ut(vec)
        singulars = self._singulars.tile(repeat_times=[1, 3]).reshape(-1)
        factors = 1.0 / (singulars + eps_reg)
        factors[singulars == 0] = 0.0
        temp[:, : tuple(singulars.shape)[0]] = (
            temp[:, : tuple(singulars.shape)[0]] * factors
        )
        return self.V(self.add_zeros(temp))


class Deblurring2D(A_functions):
    def mat_by_img(self, M, v):
        return paddle.matmul(
            x=M,
            y=v.reshape(tuple(v.shape)[0] * self.channels, self.img_dim, self.img_dim),
        ).reshape(tuple(v.shape)[0], self.channels, tuple(M.shape)[0], self.img_dim)

    def img_by_mat(self, v, M):
        return paddle.matmul(
            x=v.reshape(tuple(v.shape)[0] * self.channels, self.img_dim, self.img_dim),
            y=M,
        ).reshape(tuple(v.shape)[0], self.channels, self.img_dim, tuple(M.shape)[1])

    def __init__(self, kernel1, kernel2, channels, img_dim, device):
        self.img_dim = img_dim
        self.channels = channels
        A_small1 = paddle.zeros(shape=[img_dim, img_dim])
        for i in range(img_dim):
            for j in range(
                i - tuple(kernel1.shape)[0] // 2, i + tuple(kernel1.shape)[0] // 2
            ):
                if j < 0 or j >= img_dim:
                    continue
                A_small1[i, j] = kernel1[j - i + tuple(kernel1.shape)[0] // 2]
        A_small2 = paddle.zeros(shape=[img_dim, img_dim])
        for i in range(img_dim):
            for j in range(
                i - tuple(kernel2.shape)[0] // 2, i + tuple(kernel2.shape)[0] // 2
            ):
                if j < 0 or j >= img_dim:
                    continue
                A_small2[i, j] = kernel2[j - i + tuple(kernel2.shape)[0] // 2]
        tmp_u, tmp_s, tmp_v = paddle.linalg.svd(x=A_small1, full_matrices=not False)
        self.U_small1, self.singulars_small1, self.V_small1 = (
            tmp_u,
            tmp_s,
            tmp_v.conj().t(),
        )
        tmp_u, tmp_s, tmp_v = paddle.linalg.svd(x=A_small2, full_matrices=not False)
        self.U_small2, self.singulars_small2, self.V_small2 = (
            tmp_u,
            tmp_s,
            tmp_v.conj().t(),
        )
        ZERO = 0.03
        self.singulars_small1[self.singulars_small1 < ZERO] = 0
        self.singulars_small2[self.singulars_small2 < ZERO] = 0
        self._singulars = paddle.matmul(
            x=self.singulars_small1.reshape(img_dim, 1),
            y=self.singulars_small2.reshape(1, img_dim),
        ).reshape(img_dim**2)
        self._singulars, self._perm = paddle.sort(
            descending=True, x=self._singulars
        ), paddle.argsort(descending=True, x=self._singulars)

    def V(self, vec):
        temp = paddle.zeros(
            shape=[tuple(vec.shape)[0], self.img_dim**2, self.channels]
        )
        temp[:, self._perm, :] = vec.clone().reshape(
            tuple(vec.shape)[0], self.img_dim**2, self.channels
        )
        temp = temp.transpose(perm=[0, 2, 1])
        out = self.mat_by_img(self.V_small1, temp)
        out = self.img_by_mat(
            out, self.V_small2.transpose([1,0])
        ).reshape(tuple(vec.shape)[0], -1)
        return out

    def Vt(self, vec):
        temp = self.mat_by_img(
            self.V_small1.transpose([1,0]),
            vec.clone(),
        )
        temp = self.img_by_mat(temp, self.V_small2).reshape(
            tuple(vec.shape)[0], self.channels, -1
        )
        temp = temp[:, :, self._perm].transpose(perm=[0, 2, 1])
        return temp.reshape(tuple(vec.shape)[0], -1)

    def U(self, vec):
        temp = paddle.zeros(
            shape=[tuple(vec.shape)[0], self.img_dim**2, self.channels]
        )
        temp[:, self._perm, :] = vec.clone().reshape(
            tuple(vec.shape)[0], self.img_dim**2, self.channels
        )
        temp = temp.transpose(perm=[0, 2, 1])
        out = self.mat_by_img(self.U_small1, temp)
        out = self.img_by_mat(
            out, self.U_small2.transpose([1,0])
        ).reshape(tuple(vec.shape)[0], -1)
        return out

    def Ut(self, vec):
        temp = self.mat_by_img(
            self.U_small1.transpose([1,0]),
            vec.clone(),
        )
        temp = self.img_by_mat(temp, self.U_small2).reshape(
            tuple(vec.shape)[0], self.channels, -1
        )
        temp = temp[:, :, self._perm].transpose(perm=[0, 2, 1])
        return temp.reshape(tuple(vec.shape)[0], -1)

    def singulars(self):
        return self._singulars.tile(repeat_times=[1, 3]).reshape(-1)

    def add_zeros(self, vec):
        return vec.clone().reshape(tuple(vec.shape)[0], -1)
