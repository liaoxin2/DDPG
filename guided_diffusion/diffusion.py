import glob
import logging
import os
import random
import time

#import lpips
import numpy as np
import tqdm
from PIL import Image

import paddle
from datasets import get_dataset, data_transform, inverse_data_transform
from functions.ckpt_util import download, get_ckpt_path
from functions.ddpg_scheme import ddpg_diffusion
from guided_diffusion.models import Model
from guided_diffusion.script_util import (args_to_dict, classifier_defaults,
                                          create_classifier, create_model)
import matplotlib.pyplot as plt

#loss_fn_alex = lpips.LPIPS(net="alex")


def save_image(tensor, filename):

    array = (tensor * 255).clip(0, 255).astype("uint8").numpy()

    if array.shape[0] == 1:
        array = array.squeeze(0)
    else:
        array = array.transpose(1, 2, 0)

    image = Image.fromarray(array)
    image.save(filename)


def save_sst_images(config, args, imgs, filename, title):
    imgs = inverse_data_transform(config, imgs)
    imgs = imgs*(35.666367+2.0826182)-2.0826182
    input_np = imgs.squeeze().cpu().numpy()
    plt.imshow(input_np, cmap='hot')
    plt.colorbar(label='sst')
    plt.title(title)
    plt.savefig(os.path.join(args.image_folder, filename), bbox_inches='tight', dpi=300)
    plt.close()

                         
def get_gaussian_noisy_img(img, noise_level):
    return (
        img
        + paddle.randn(shape=img.shape, dtype=img.dtype).cuda(blocking=True)
        * noise_level
    )


def MeanUpsample(x, scale):
    n, c, h, w = tuple(x.shape)
    out = paddle.zeros(shape=[n, c, h, scale, w, scale]).to(x.place) + x.view(
        n, c, h, 1, w, 1
    )
    out = out.view(n, c, scale * h, scale * w)
    return out


def color2gray(x):
    coef = 1 / 3
    x = x[:, 0, :, :] * coef + x[:, 1, :, :] * coef + x[:, 2, :, :] * coef
    return x.tile(repeat_times=[1, 3, 1, 1])


def gray2color(x):
    x = x[:, 0, :, :]
    coef = 1 / 3
    base = coef**2 + coef**2 + coef**2
    return paddle.stack(x=(x * coef / base, x * coef / base, x * coef / base), axis=1)


def get_beta_schedule(beta_schedule, *, beta_start, beta_end, num_diffusion_timesteps):
    def sigmoid(x):
        return 1 / (np.exp(-x) + 1)

    if beta_schedule == "quad":
        betas = (
            np.linspace(
                beta_start**0.5,
                beta_end**0.5,
                num_diffusion_timesteps,
                dtype=np.float64,
            )
            ** 2
        )
    elif beta_schedule == "linear":
        betas = np.linspace(
            beta_start, beta_end, num_diffusion_timesteps, dtype=np.float64
        )
    elif beta_schedule == "const":
        betas = beta_end * np.ones(num_diffusion_timesteps, dtype=np.float64)
    elif beta_schedule == "jsd":
        betas = 1.0 / np.linspace(
            num_diffusion_timesteps, 1, num_diffusion_timesteps, dtype=np.float64
        )
    elif beta_schedule == "sigmoid":
        betas = np.linspace(-6, 6, num_diffusion_timesteps)
        betas = sigmoid(betas) * (beta_end - beta_start) + beta_start
    else:
        raise NotImplementedError(beta_schedule)
    assert tuple(betas.shape) == (num_diffusion_timesteps,)
    return betas


class Diffusion(object):
    def __init__(self, args, config, device=None):
        self.args = args
        self.config = config
        if device is None:
            device = (
                str("cuda").replace("cuda", "gpu")
                if paddle.device.cuda.device_count() >= 1
                else paddle.CPUPlace()
            )
        self.device = device
        self.model_var_type = config.model.var_type
        betas = get_beta_schedule(
            beta_schedule=config.diffusion.beta_schedule,
            beta_start=config.diffusion.beta_start,
            beta_end=config.diffusion.beta_end,
            num_diffusion_timesteps=config.diffusion.num_diffusion_timesteps,
        )
        betas = self.betas = (
            paddle.to_tensor(data=betas).astype(dtype="float32").to(self.device)
        )
        self.num_timesteps = tuple(betas.shape)[0]
        alphas = 1.0 - betas
        alphas_cumprod = alphas.cumprod(dim=0)
        alphas_cumprod_prev = paddle.concat(
            x=[paddle.ones(shape=[1]).to(device), alphas_cumprod[:-1]], axis=0
        )
        self.alphas_cumprod_prev = alphas_cumprod_prev
        posterior_variance = (
            betas * (1.0 - alphas_cumprod_prev) / (1.0 - alphas_cumprod)
        )
        if self.model_var_type == "fixedlarge":
            self.logvar = betas.log()
        elif self.model_var_type == "fixedsmall":
            self.logvar = posterior_variance.clip(min=1e-20).log()

    def sample(self, logger):
        cls_fn = None
        if self.config.model.type == "simple":
            model = Model(self.config)
            if self.config.data.dataset == "CIFAR10":
                name = "cifar10"
            elif self.config.data.dataset == "LSUN":
                name = f"lsun_{self.config.data.category}"
            elif self.config.data.dataset == "CelebA_HQ":
                name = "celeba_hq"
            elif self.config.data.dataset == "sst":
                name = "sst"
            else:
                raise ValueError
            if name != "celeba_hq" and name != 'sst':
                ckpt = get_ckpt_path(f"ema_{name}", prefix=self.args.exp)
                print("Loading checkpoint {}".format(ckpt))
            elif name == "celeba_hq":
                ckpt = os.path.join(self.args.exp, "logs/celeba/celeba_hq.pdparams")
                if not os.path.exists(ckpt):
                    raise ValueError(
                        "CelebA-HQ model checkpoint not found, please download it as mentioned in README.md file and configure the correct path"
                    )
            elif name == 'sst':
                ckpt = self.config.sampling.pretrained_model_path
            else:
                raise ValueError
            model.set_state_dict(state_dict=paddle.load(path=str(ckpt)))
            model.to(self.device)
            model = paddle.DataParallel(layers=model)
        elif self.config.model.type == "openai":
            config_dict = vars(self.config.model)
            model = create_model(**config_dict)
            if self.config.model.use_fp16:
                model.convert_to_fp16()
            if self.config.model.class_cond:
                ckpt = os.path.join(
                    self.args.exp,
                    "logs/imagenet/%dx%d_diffusion.pt"
                    % (self.config.data.image_size, self.config.data.image_size),
                )
                if not os.path.exists(ckpt):
                    raise ValueError(
                        "ImageNet model checkpoint not found, please download it as mentioned in README.md file and configure the correct path"
                    )
            else:
                ckpt = os.path.join(
                    self.args.exp, "logs/imagenet/256x256_diffusion_uncond.pt"
                )
                if not os.path.exists(ckpt):
                    raise ValueError(
                        "ImageNet model checkpoint not found, please download it as mentioned in README.md file and configure the correct path"
                    )
            model.set_state_dict(state_dict=paddle.load(path=str(ckpt)))
            model.to(self.device)
            model.eval()
            model = paddle.DataParallel(layers=model)
            if self.config.model.class_cond:
                ckpt = os.path.join(
                    self.args.exp,
                    "logs/imagenet/%dx%d_classifier.pt"
                    % (self.config.data.image_size, self.config.data.image_size),
                )
                if not os.path.exists(ckpt):
                    image_size = self.config.data.image_size
                    download(
                        "https://openaipublic.blob.core.windows.net/diffusion/jul-2021/%dx%d_classifier.pt"
                        % image_size,
                        ckpt,
                    )
                classifier = create_classifier(
                    **args_to_dict(self.config.classifier, classifier_defaults().keys())
                )
                classifier.set_state_dict(state_dict=paddle.load(path=str(ckpt)))
                classifier.to(self.device)
                if self.config.classifier.classifier_use_fp16:
                    classifier.convert_to_fp16()
                classifier.eval()
                classifier = paddle.DataParallel(layers=classifier)
                pass

                def cond_fn(x, t, y):
                    with paddle.enable_grad():
                        out_1 = x.detach()
                        out_1.stop_gradient = not True
                        x_in = out_1
                        logits = classifier(x_in, t)
                        log_probs = paddle.nn.functional.log_softmax(x=logits, axis=-1)
                        selected = log_probs[range(len(logits)), y.view(-1)]
                        return (
                            paddle.grad(outputs=selected.sum(), inputs=x_in)[0]
                            * self.config.classifier.classifier_scale
                        )

                cls_fn = cond_fn
        if self.args.inject_noise == 1:
            print(
                "Run DDPG.",
                f"Operators implementation via {self.args.operator_imp}.",
                f"{self.config.sampling.T_sampling} sampling steps.",
                f"Task: {self.args.deg}.",
                f"Noise level: {self.args.sigma_y}.",
            )
        else:
            print(
                "Run IDPG.",
                f"Operators implementation via {self.args.operator_imp}.",
                f"{self.config.sampling.T_sampling} sampling steps.",
                f"Task: {self.args.deg}.",
                f"Noise level: {self.args.sigma_y}.",
            )
        self.ddpg_wrapper(model, cls_fn, logger)

    def ddpg_wrapper(self, model, cls_fn, logger):
        args, config = self.args, self.config
        dataset, test_dataset = get_dataset(args, config)
        device_count = paddle.device.cuda.device_count()
        if args.subset_start >= 0 and args.subset_end > 0:
            assert args.subset_end > args.subset_start
            test_dataset = paddle.io.Subset(
                dataset=test_dataset, indices=range(args.subset_start, args.subset_end)
            )
        else:
            args.subset_start = 0
            args.subset_end = len(test_dataset)
        print(f"Dataset has size {len(test_dataset)}")

        def seed_worker(worker_id):
            worker_seed = args.seed % 2**32
            np.random.seed(worker_seed)
            random.seed(worker_seed)

        g = paddle.framework.core.default_cpu_generator()
        g.manual_seed(args.seed)
        val_loader = paddle.io.DataLoader(
            dataset=test_dataset,
            batch_size=config.sampling.batch_size,
            shuffle=False,
            num_workers=config.data.num_workers,
            worker_init_fn=seed_worker,
        )
        deg = args.deg
        A_funcs = None
        if deg == "cs_walshhadamard":
            compress_by = round(1 / args.deg_scale)
            from functions.svd_operators import WalshHadamardCS

            A_funcs = WalshHadamardCS(
                config.data.channels,
                self.config.data.image_size,
                compress_by,
                paddle.randperm(n=self.config.data.image_size**2),
                self.device,
            )
        elif deg == "cs_blockbased":
            cs_ratio = args.deg_scale
            from functions.svd_operators import CS

            A_funcs = CS(
                config.data.channels, self.config.data.image_size, cs_ratio, self.device
            )
        elif deg == "inpainting":
            from functions.svd_operators import Inpainting

            loaded = np.load("exp/inp_masks/mask.npy")
            mask = paddle.to_tensor(data=loaded).to(self.device).reshape(-1)
            missing_r = (
                paddle.nonzero(x=mask == 0).astype(dtype="int64").reshape(-1) * 3
            )
            missing_g = missing_r + 1
            missing_b = missing_g + 1
            missing = paddle.concat(x=[missing_r, missing_g, missing_b], axis=0)
            A_funcs = Inpainting(
                config.data.channels, config.data.image_size, missing, self.device
            )
        elif deg == "denoising":
            from functions.svd_operators import Denoising

            A_funcs = Denoising(
                config.data.channels, self.config.data.image_size, self.device
            )
        elif deg == "colorization":
            from functions.svd_operators import Colorization

            A_funcs = Colorization(config.data.image_size, self.device)
        elif deg == "sr_averagepooling":
            blur_by = int(args.deg_scale)
            if args.operator_imp == "SVD":
                from functions.svd_operators import SuperResolution

                A_funcs = SuperResolution(
                    config.data.channels, config.data.image_size, blur_by, self.device
                )
            else:
                raise NotImplementedError()
        elif deg == "sr_bicubic":
            factor = int(args.deg_scale)

            def bicubic_kernel(x, a=-0.5):
                if abs(x) <= 1:
                    return (a + 2) * abs(x) ** 3 - (a + 3) * abs(x) ** 2 + 1
                elif 1 < abs(x) and abs(x) < 2:
                    return (
                        a * abs(x) ** 3 - 5 * a * abs(x) ** 2 + 8 * a * abs(x) - 4 * a
                    )
                else:
                    return 0

            k = np.zeros(factor * 4)
            for i in range(factor * 4):
                x = 1 / factor * (i - np.floor(factor * 4 / 2) + 0.5)
                k[i] = bicubic_kernel(x)
            k = k / np.sum(k)
            kernel = paddle.to_tensor(data=k).astype(dtype="float32").to(self.device)
            if args.operator_imp == "SVD":
                from functions.svd_operators import SRConv

                A_funcs = SRConv(
                    kernel / kernel.sum(),
                    config.data.channels,
                    self.config.data.image_size,
                    self.device,
                    stride=factor,
                )
            elif args.operator_imp == "SVD_sst":
                from functions.svd_operators import SRConv_NonSquare
                A_funcs = SRConv_NonSquare(
                    kernel / kernel.sum(), 
                    config.data.channels, 
                    self.config.data.image_lat, 
                    self.config.data.image_lon, 
                    self.device, 
                    stride=factor,
                )
            elif args.operator_imp == "FFT":
                from functions.fft_operators import (Superres_fft,
                                                     prepare_cubic_filter)

                k = prepare_cubic_filter(1 / factor)
                kernel = (
                    paddle.to_tensor(data=k).astype(dtype="float32").to(self.device)
                )
                A_funcs = Superres_fft(
                    kernel / kernel.sum(),
                    config.data.channels,
                    self.config.data.image_size,
                    self.device,
                    stride=factor,
                )
            else:
                raise NotImplementedError()
        elif deg == "deblur_uni":
            if args.operator_imp == "SVD":
                from functions.svd_operators import Deblurring

                A_funcs = Deblurring(
                    paddle.to_tensor(data=[1 / 9] * 9, dtype="float32").to(self.device),
                    config.data.channels,
                    self.config.data.image_size,
                    self.device,
                )
            elif args.operator_imp == "FFT":
                from functions.fft_operators import Deblurring_fft

                A_funcs = Deblurring_fft(
                    paddle.to_tensor(data=[1 / 9] * 9, dtype="float32").to(self.device),
                    config.data.channels,
                    self.config.data.image_size,
                    self.device,
                )
            else:
                raise NotImplementedError()
        elif deg == "deblur_gauss":
            sigma = 10
            pdf = lambda x: paddle.exp(
                x=paddle.to_tensor(data=[-0.5 * (x / sigma) ** 2], dtype="float32")
            )
            kernel = paddle.to_tensor(
                data=[pdf(-2), pdf(-1), pdf(0), pdf(1), pdf(2)], dtype="float32"
            ).to(self.device)
            if args.operator_imp == "SVD":
                from functions.svd_operators import Deblurring

                A_funcs = Deblurring(
                    kernel / kernel.sum(),
                    config.data.channels,
                    self.config.data.image_size,
                    self.device,
                )
            elif args.operator_imp == "FFT":
                from functions.fft_operators import Deblurring_fft

                A_funcs = Deblurring_fft(
                    kernel / kernel.sum(),
                    config.data.channels,
                    self.config.data.image_size,
                    self.device,
                )
            else:
                raise NotImplementedError()
        elif deg == "deblur_aniso":
            sigma = 20
            pdf = lambda x: paddle.exp(
                x=paddle.to_tensor(data=[-0.5 * (x / sigma) ** 2], dtype="float32")
            )
            kernel2 = paddle.to_tensor(
                data=[
                    pdf(-4),
                    pdf(-3),
                    pdf(-2),
                    pdf(-1),
                    pdf(0),
                    pdf(1),
                    pdf(2),
                    pdf(3),
                    pdf(4),
                ],
                dtype="float32",
            ).to(self.device)
            sigma = 1
            pdf = lambda x: paddle.exp(
                x=paddle.to_tensor(data=[-0.5 * (x / sigma) ** 2], dtype="float32")
            )
            kernel1 = paddle.to_tensor(
                data=[
                    pdf(-4),
                    pdf(-3),
                    pdf(-2),
                    pdf(-1),
                    pdf(0),
                    pdf(1),
                    pdf(2),
                    pdf(3),
                    pdf(4),
                ],
                dtype="float32",
            ).to(self.device)
            if args.operator_imp == "SVD":
                from functions.svd_operators import Deblurring2D

                A_funcs = Deblurring2D(
                    kernel1 / kernel1.sum(),
                    kernel2 / kernel2.sum(),
                    config.data.channels,
                    self.config.data.image_size,
                    self.device,
                )
            elif args.operator_imp == "FFT":
                from functions.fft_operators import Deblurring_fft

                kernel = paddle.matmul(x=kernel1[:, None], y=kernel2[None, :])
                A_funcs = Deblurring_fft(
                    kernel / kernel.sum(),
                    config.data.channels,
                    self.config.data.image_size,
                    self.device,
                )
            else:
                raise NotImplementedError()
        elif deg == "motion_deblur":
            from functions.motionblur import Kernel

            if args.operator_imp == "FFT":
                from functions.fft_operators import Deblurring_fft
            else:
                raise ValueError("set operator_imp = FFT")
        else:
            raise ValueError("degradation type not supported")
        args.sigma_y = 2 * args.sigma_y
        sigma_y = args.sigma_y
        print(f"Start from {args.subset_start}")
        idx_init = args.subset_start
        idx_so_far = args.subset_start
        avg_psnr = 0.0
        avg_lpips = 0.0
        pbar = tqdm.tqdm(enumerate(val_loader))
        img_ind = -1
        for classes, x_orig in pbar:
            img_ind = img_ind + 1
            if deg == "motion_deblur":
                np.random.seed(seed=img_ind * 10)
                kernel = paddle.to_tensor(
                    data=Kernel(size=(61, 61), intensity=0.5).kernelMatrix
                )
                A_funcs = Deblurring_fft(
                    kernel / kernel.sum(),
                    config.data.channels,
                    self.config.data.image_size,
                    self.device,
                )
                np.random.seed(seed=args.seed)
            if self.config.data.dataset == "sst":
                target = x_orig[1].to(self.device)
                target = data_transform(self.config, target)
            x_orig = x_orig[0].to(self.device)
            x_orig = data_transform(self.config, x_orig)
            
 
            y = A_funcs.A(x_orig)
            y = y + args.sigma_y * paddle.randn(shape=y.shape, dtype=y.dtype).cuda(
                blocking=True
            )

            b, hwc = tuple(y.shape)
            if "color" in deg:
                hw = hwc / 1
                h = w = int(hw**0.5)
                y = y.reshape((b, 1, h, w))
            elif "inp" in deg or "cs" in deg:
                pass
            elif args.operator_imp == "SVD_sst":
                h = self.config.data.image_lat//args.deg_scale
                w = self.config.data.image_lon//args.deg_scale
                y = y.reshape([b, 1, int(h), int(w)])
            else:
                hw = hwc / 3
                h = w = int(hw**0.5)
                y = y.reshape((b, 3, h, w))
            y = y.reshape((b, hwc))
            if self.config.data.dataset == "sst":
                Apy = A_funcs.A_pinv_add_eta(
                    y, max(0.0001, sigma_y**2 * args.eta_tilde)
                ).view(
                    [y.shape[0],
                    config.data.channels,
                    self.config.data.image_lat,
                    self.config.data.image_lon]
                )
            else:
                Apy = A_funcs.A_pinv_add_eta(
                    y, max(0.0001, sigma_y**2 * args.eta_tilde)
                ).view(
                    [y.shape[0],
                    config.data.channels,
                    self.config.data.image_size,
                    self.config.data.image_size]
                )
            if args.save_observed_img:                 
                os.makedirs(os.path.join(self.args.image_folder, "Apy"), exist_ok=True)
                for i in range(len(Apy)):
                    if self.config.data.dataset == "sst":
                        save_sst_images(config, args, y[i].reshape([b, 1, int(h), int(w)]), f"Apy/y_{idx_so_far + i}.png", 'Downsample the forecasted sea surface temperature by a factor of 16')
                        save_sst_images(config, args, x_orig[i], f"Apy/orig_{idx_so_far + i}.png", 'Forecasted sea surface temperature')
                        save_sst_images(config, args, target[i], f"Apy/traget_{idx_so_far + i}.png", 'Observed sea surface temperature')
                    else:
                        save_image(
                            inverse_data_transform(config, Apy[i]),
                            os.path.join(
                                self.args.image_folder, f"Apy/Apy_{idx_so_far + i}.png"
                            ),
                        )
                        save_image(
                            inverse_data_transform(config, x_orig[i]),
                            os.path.join(
                                self.args.image_folder, f"Apy/orig_{idx_so_far + i}.png"
                            ),
                        )
                        if "inp" in deg or "cs" in deg:
                            pass
                        else:
                            save_image(
                                inverse_data_transform(
                                    config, y[i].reshape((3, h, w))
                                ),
                                os.path.join(
                                    self.args.image_folder, f"Apy/y_{idx_so_far + i}.png"
                                ),
                            )
            if self.config.data.dataset == "sst":
                x = paddle.randn(
                    shape=[
                        tuple(y.shape)[0],
                        config.data.channels,
                        config.data.image_lat,
                        config.data.image_lon,
                    ]
                )
            else:
                x = paddle.randn(
                    shape=[
                        tuple(y.shape)[0],
                        config.data.channels,
                        config.data.image_size,
                        config.data.image_size,
                    ]
                )
            with paddle.no_grad():
                x, _ = ddpg_diffusion(
                    x,
                    model,
                    self.betas,
                    A_funcs,
                    y,
                    sigma_y,
                    cls_fn=cls_fn,
                    classes=classes,
                    config=config,
                    args=args,
                )
            lpips_final = 0 #(
            #     paddle.squeeze(x=loss_fn_alex(x[0], x_orig.to("cpu"))).detach().numpy()
            # )
            avg_lpips += lpips_final

            x = [inverse_data_transform(config, xi) for xi in x]
            for j in range(x[0].shape[0]):
                if self.config.data.dataset == "sst":
                    save_sst_images(config, args, x[0][j], f"{idx_so_far + j}_0.png", 'Predicted sea surface temperature')

                    target_i = inverse_data_transform(config, target[0])   
                    x_pred = x[0][j]
                    yubao = inverse_data_transform(config, x_orig[0])
                    x_pred = x_pred*(35.666367+2.0826182)-2.0826182
                    target_i = target_i*(35.666367+2.0826182)-2.0826182
                    yubao = yubao*(35.666367+2.0826182)-2.0826182
                    input_np = x_pred.squeeze().cpu().numpy()  
                    target_np = target_i.squeeze().cpu().numpy()  
                    yubao_np = yubao.squeeze().cpu().numpy() 

                    error = np.abs(target_np-input_np)
                    plt.imshow(error, cmap='hot')
                    plt.colorbar(label='sst')
                    plt.title("Absolute error of sea surface temperature")
                    plt.savefig(os.path.join(self.args.image_folder, f"error_{idx_so_far + j}_{0}.png"), bbox_inches='tight', dpi=300)
                    plt.close()

                    error = np.abs(target_np-yubao_np)
                    plt.imshow(error, cmap='hot')
                    plt.colorbar(label='sst')
                    plt.title("Absolute error of sea surface temperature")
                    plt.savefig(os.path.join(self.args.image_folder, f"error_yubao_guance_{idx_so_far + j}_{0}.png"), bbox_inches='tight', dpi=300)
                    plt.close()

                    error = np.abs(yubao_np-input_np)
                    plt.imshow(error, cmap='hot')
                    plt.colorbar(label='sst')
                    plt.title("Absolute error of sea surface temperature")
                    plt.savefig(os.path.join(self.args.image_folder, f"error_yubao_pred_{idx_so_far + j}_{0}.png"), bbox_inches='tight', dpi=300)
                    plt.close()
                else:
                    save_image(
                        x[0][j],
                        os.path.join(self.args.image_folder, f"{idx_so_far + j}_{0}.png"),
                    )
                if self.config.data.dataset == "sst":
                    target = inverse_data_transform(config, target[j])
                else:
                    target = inverse_data_transform(config, x_orig[j])
                mse = paddle.mean(x=(x[0][j].to(self.device) - target) ** 2)
                psnr = 10 * paddle.log10(x=1 / mse)
                logger.info(
                    "img_ind: %d, PSNR: %.2f, LPIPS: %.4f"
                    % (img_ind, psnr, lpips_final)
                )
                avg_psnr += psnr
            idx_so_far += tuple(y.shape)[0]
            logger.info(
                "Avg PSNR: %.2f, Avg LPIPS: %.4f     (** After %d iteration **)"
                % (
                    avg_psnr / (idx_so_far - idx_init),
                    avg_lpips / (idx_so_far - idx_init),
                    idx_so_far - idx_init,
                )
            )
        avg_psnr = avg_psnr / (idx_so_far - idx_init)
        avg_lpips = avg_lpips / (idx_so_far - idx_init)
        print("Total Average PSNR: %.2f" % avg_psnr)
        print("Total Average LPIPS: %.4f" % avg_lpips)
        print("Number of samples: %d" % (idx_so_far - idx_init))
