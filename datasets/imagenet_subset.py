from PIL import Image

import paddle


class CenterCropLongEdge(object):
    """Crops the given PIL Image on the long edge.
    Args:
        size (sequence or int): Desired output size of the crop. If size is an
            int instead of sequence like (h, w), a square crop (size, size) is
            made.
    """

    def __call__(self, img):
        """
        Args:
            img (PIL Image): Image to be cropped.
        Returns:
            PIL Image: Cropped image.
        """
        return paddle.vision.transforms.center_crop(img=img, output_size=min(img.size))

    def __repr__(self):
        return self.__class__.__name__


def pil_loader(path):
    with open(path, "rb") as f:
        img = Image.open(f)
        return img.convert("RGB")


def accimage_loader(path):
    import accimage

    try:
        return accimage.Image(path)
    except IOError:
        return pil_loader(path)


def default_loader(path):
>>>>>>    if torchvision.get_image_backend() == "accimage":
        return accimage_loader(path)
    else:
        return pil_loader(path)


class ImageDataset(paddle.io.Dataset):
    def __init__(
        self, root_dir, meta_file, transform=None, image_size=128, normalize=True
    ):
        self.root_dir = root_dir
        if transform is not None:
            self.transform = transform
        else:
            norm_mean = [0.5, 0.5, 0.5]
            norm_std = [0.5, 0.5, 0.5]
            if normalize:
                self.transform = paddle.vision.transforms.Compose(
                    transforms=[
                        CenterCropLongEdge(),
                        paddle.vision.transforms.Resize(size=image_size),
                        paddle.vision.transforms.ToTensor(),
                        paddle.vision.transforms.Normalize(
                            mean=norm_mean, std=norm_std
                        ),
                    ]
                )
            else:
                self.transform = paddle.vision.transforms.Compose(
                    transforms=[
                        CenterCropLongEdge(),
                        paddle.vision.transforms.Resize(size=image_size),
                        paddle.vision.transforms.ToTensor(),
                    ]
                )
        with open(meta_file) as f:
            lines = f.readlines()
        print("building dataset from %s" % meta_file)
        self.num = len(lines)
        self.metas = []
        self.classifier = None
        suffix = ""
        for line in lines:
            line_split = line.rstrip().split()
            if len(line_split) == 2:
                self.metas.append((line_split[0] + suffix, int(line_split[1])))
            else:
                self.metas.append((line_split[0] + suffix, -1))
        print("read meta done")

    def __len__(self):
        return self.num

    def __getitem__(self, idx):
        filename = self.root_dir + "/" + self.metas[idx][0]
        cls = self.metas[idx][1]
        img = default_loader(filename)
        if self.transform is not None:
            img = self.transform(img)
        return img, cls
