import os

import numpy as np
from tqdm import tqdm

import paddle

class_num = 951


def compute_alpha(beta, t):
    beta = paddle.concat(x=[paddle.zeros(shape=[1]).to(beta.place), beta], axis=0)
    a = (1 - beta).cumprod(dim=0).index_select(axis=0, index=t + 1).view(-1, 1, 1, 1)
    return a


def ddpg_diffusion(
    x, model, b, A_funcs, y, sigma_y, cls_fn=None, classes=None, config=None, args=None
):
    with paddle.no_grad():
        skip = config.diffusion.num_diffusion_timesteps // config.sampling.T_sampling
        n = x.shape[0]
        x0_preds = []
        xs = [x]
        times = get_schedule_jump(config.sampling.T_sampling, 1, 1)
        time_pairs = list(zip(times[:-1], times[1:]))
        for i, j in tqdm(time_pairs):
            i, j = i * skip, j * skip
            if j < 0:
                j = -1
            if j < i:
                t = (paddle.ones(shape=n) * i).to(x.place)
                next_t = (paddle.ones(shape=n) * j).to(x.place)
                at = compute_alpha(b, t.astype(dtype="int64"))
                at_next = compute_alpha(b, next_t.astype(dtype="int64"))
                xt = xs[-1].to("gpu")
                if cls_fn == None:
                    et = model(xt, t)
                else:
                    classes = paddle.ones(shape=xt.shape[0], dtype="int64") * class_num
                    et = model(xt, t, classes)
                    et = et[:, :3]
                    et = et - (1 - at).sqrt()[0, 0, 0, 0] * cls_fn(x, t, classes)
                if et.shape[1] == 6:
                    et = et[:, :3]
                x0_t = (xt - et * (1 - at).sqrt()) / at.sqrt()
                if sigma_y == 0.0:
                    delta_t = 0
                    weight_noise_t = 1
                else:
                    delta_t = at_next**args.gamma
                    weight_noise_t = delta_t
                eta_reg = max(0.0001, sigma_y**2 * args.eta_tilde)
                if args.eta_tilde < 0:
                    eta_reg = 0.0001 + args.xi * (sigma_y * 255.0) ** 2
                scale_gLS = args.scale_ls
                guidance_BP = A_funcs.A_pinv_add_eta(
                    A_funcs.A(x0_t.reshape(x0_t.shape[0], -1))
                    - y.reshape(y.shape[0], -1),
                    eta_reg,
                ).reshape(*tuple(x0_t.shape))
                guidance_LS = A_funcs.At(
                    A_funcs.A(x0_t.reshape(x0_t.shape[0], -1))
                    - y.reshape(y.shape[0], -1)
                ).reshape(*tuple(x0_t.shape))
                if args.step_size_mode == 0:
                    step_size_LS = 1
                    step_size_BP = 1
                    step_size = 1
                elif args.step_size_mode == 1:
                    step_size_LS = 1
                    step_size_BP = 1
                    step_size = (1 - at_next) / (1 - at)
                elif args.step_size_mode == 2:
                    step_size_LS = (1 - at_next) / (1 - at)
                    step_size_BP = 1
                    step_size = 1
                else:
                    assert 1, "unsupported step-size mode"
                xt_next_tilde = x0_t - step_size * (
                    step_size_BP * (1 - delta_t) * guidance_BP
                    + step_size_LS * delta_t * scale_gLS * guidance_LS
                )
                et_hat = (xt - at.sqrt() * xt_next_tilde) / (1 - at).sqrt()
                c1 = 0
                c2 = 0
                if args.inject_noise:
                    zeta = args.zeta
                    c1 = (1 - at_next).sqrt() * np.sqrt(zeta)
                    c2 = (1 - at_next).sqrt() * np.sqrt(1 - zeta) * weight_noise_t
                xt_next = (
                    at_next.sqrt() * xt_next_tilde
                    + c1 * paddle.randn(shape=x0_t.shape, dtype=x0_t.dtype)
                    + c2 * et_hat
                )
                x0_preds.append(x0_t.to("cpu"))
                xs.append(xt_next.to("cpu"))
            else:
                assert 1, "Unexpected case"
        if sigma_y != 0.0:
            xs.append(x0_t.to("cpu"))
    return [xs[-1]], [x0_preds[-1]]


def get_schedule_jump(T_sampling, travel_length, travel_repeat):
    jumps = {}
    for j in range(0, T_sampling - travel_length, travel_length):
        jumps[j] = travel_repeat - 1
    t = T_sampling
    ts = []
    while t >= 1:
        t = t - 1
        ts.append(t)
        if jumps.get(t, 0) > 0:
            jumps[t] = jumps[t] - 1
            for _ in range(travel_length):
                t = t + 1
                ts.append(t)
    ts.append(-1)
    _check_times(ts, -1, T_sampling)
    return ts


def _check_times(times, t_0, T_sampling):
    assert times[0] > times[1], (times[0], times[1])
    assert times[-1] == -1, times[-1]
    for t_last, t_cur in zip(times[:-1], times[1:]):
        assert abs(t_last - t_cur) == 1, (t_last, t_cur)
    for t in times:
        assert t >= t_0, (t, t_0)
        assert t <= T_sampling, (t, T_sampling)
