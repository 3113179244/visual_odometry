"""
this file is modified from https://github.com/leeyeehoo/CSRNet-pytorch
to make it compatible with python3, and support wts file exporting
"""

import struct

import cv2
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import models


class CSRNet(nn.Module):
    def __init__(self, load_weights=False):
        super(CSRNet, self).__init__()
        self.seen = 0
        self.frontend_feat = [64, 64, "M", 128, 128, "M", 256, 256, 256, "M", 512, 512, 512]
        self.backend_feat = [512, 512, 512, 256, 128, 64]
        self.frontend = make_layers(self.frontend_feat)
        self.backend = make_layers(self.backend_feat, in_channels=512, dilation=True)
        self.output_layer = nn.Conv2d(64, 1, kernel_size=1)
        if not load_weights:
            mod = models.vgg16(pretrained=True)
            self._initialize_weights()
            for i in range(len(self.frontend.state_dict().items())):
                self.frontend.state_dict().items()[i][1].data[:] = mod.state_dict().items()[i][1].data[:]

    def forward(self, x):
        x = self.frontend(x)
        x = self.backend(x)
        x = self.output_layer(x)
        return x

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.normal_(m.weight, std=0.01)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)


def make_layers(cfg, in_channels=3, batch_norm=False, dilation=False):
    if dilation:
        d_rate = 2
    else:
        d_rate = 1
    layers = []
    for v in cfg:
        if v == "M":
            layers += [nn.MaxPool2d(kernel_size=2, stride=2)]
        else:
            conv2d = nn.Conv2d(in_channels, v, kernel_size=3, padding=d_rate, dilation=d_rate)
            if batch_norm:
                layers += [conv2d, nn.BatchNorm2d(v), nn.ReLU(inplace=True)]
            else:
                layers += [conv2d, nn.ReLU(inplace=True)]
            in_channels = v
    return nn.Sequential(*layers)


def preprocess(img: np.array) -> torch.Tensor:
    """
    a preprocess method align with CSRNet: no resize, no normalization
    input resolution: (1024, 768, 3)

    Args:
        img (np.array): input image

    Returns:
        torch.Tensor: preprocessed image in `NCHW` layout
    """
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB).astype(np.float32)
    mean = np.array([92.8207477031, 95.2757037428, 104.877445883], dtype=np.float32)
    img = img - mean
    img = img.transpose(2, 0, 1)[None, ...]
    return torch.from_numpy(img)


def visualize() -> None:
    """this method is for anyone who wants to see the ground truth output
    of the input included in this demo, just call this method directly.
    """
    import scipy

    def gaussian_filter_density(gt):
        density = np.zeros(gt.shape, dtype=np.float32)
        gt_count = np.count_nonzero(gt)

        if gt_count == 0:
            return density

        ys, xs = np.nonzero(gt)
        pts = np.column_stack((xs, ys))

        tree = scipy.spatial.cKDTree(pts, leafsize=2048)  # build KDTree
        distances, _ = tree.query(pts, k=min(4, gt_count))

        for i, (x, y) in enumerate(pts):
            pt2d = np.zeros(gt.shape, dtype=np.float32)
            pt2d[y, x] = 1.0

            if gt_count > 1:
                sigma = np.sum(distances[i][1:]) * 0.1
            else:
                sigma = np.mean(gt.shape) / 4.0

            density += scipy.ndimage.gaussian_filter(pt2d, sigma, mode="constant")

        return density

    def visualize_density_cv2(img: np.array, density):
        if density.max() > 0:
            density_norm = density / density.max()
        else:
            density_norm = density

        density_uint8 = (density_norm * 255).astype(np.uint8)
        heatmap = cv2.applyColorMap(density_uint8, cv2.COLORMAP_JET)
        blended = cv2.addWeighted(img, 0.7, heatmap, 0.3, 0)
        return blended

    img = cv2.imread("../assets/IMG_1.jpg", cv2.IMREAD_COLOR)
    gt = scipy.io.loadmat("../assets/GT_IMG_1.mat")
    gt = gt["image_info"][0, 0][0, 0][0]
    k = np.zeros((img.shape[0], img.shape[1]))
    for i in range(0, len(gt)):
        if int(gt[i][1]) < img.shape[0] and int(gt[i][0]) < img.shape[1]:
            k[int(gt[i][1]), int(gt[i][0])] = 1
    k = gaussian_filter_density(k)
    img = visualize_density_cv2(img, k)
    cv2.imwrite("../assets/csrnet_gt_vis.jpg", img)


def main():
    # load pth model
    model_path = "../models/PartAmodel_best.pth.tar"  # partBmodel_best.pth
    model = CSRNet(load_weights=True)
    checkpoint = torch.load(model_path, weights_only=False)
    model.load_state_dict(checkpoint["state_dict"])
    model.eval()
    with torch.inference_mode():
        img = cv2.imread("../assets/IMG_1.jpg", cv2.IMREAD_COLOR)
        h, w, _ = img.shape
        data = preprocess(img)
        output = model(data)

        # this is the postprocess part
        # 1. convert the output to heatmap
        heatmap = F.interpolate(output, size=(h, w), mode="bilinear", align_corners=False)
        heatmap = heatmap[0].detach().cpu().numpy().transpose(1, 2, 0)
        heatmap = cv2.normalize(heatmap, None, 0, 255, cv2.NORM_MINMAX)
        heatmap = cv2.applyColorMap(heatmap.astype(np.uint8)[..., None], cv2.COLORMAP_JET)
        # 2. blend the heatmap with the original image
        heatmap = cv2.resize(heatmap, (w, h), 0, 0, cv2.INTER_LINEAR)
        overlay = cv2.addWeighted(img, 0.7, heatmap, 0.3, 0)
        status = cv2.imwrite("../assets/csrnet_output_torch.jpg", overlay)
        assert status

    # save to wts
    print("Writing into csrnet.wts")
    with open("../models/csrnet.wts", "w") as f:
        f.write("{}\n".format(len(model.state_dict().keys())))
        for k, v in model.state_dict().items():
            print(f"key: {k}\tvalue: {v.shape}")
            vr = v.reshape(-1).cpu().numpy()
            f.write("{} {} ".format(k, len(vr)))
            for vv in vr:
                f.write(" ")
                f.write(struct.pack(">f", float(vv)).hex())
            f.write("\n")


if __name__ == "__main__":
    # Uncomment the following line to generate ground truth heatmap
    # with the file in `../assets/GT_IMG_1.mat`, check the result
    # at `../assets/csrnet_gt_vis.jpg`
    # visualize()

    main()
