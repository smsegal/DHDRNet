# based on color space code from https://github.com/cassidylaidlaw/ReColorAdv


import torch


class ColorSpace(object):
    """
    Base class for color spaces.
    """

    def from_rgb(self, imgs):
        """
        Converts an Nx3xWxH tensor in RGB color space to a Nx3xWxH tensor in
        this color space. All outputs should be in the 0-1 range.
        """
        raise NotImplementedError()

    def to_rgb(self, imgs):
        """
        Converts an Nx3xWxH tensor in this color space to a Nx3xWxH tensor in
        RGB color space.
        """
        raise NotImplementedError()


class YPbPrColorSpace(ColorSpace):
    """
    YPbPr color space. Uses ITU-R BT.601 standard by default.
    """

    def __init__(self, kr=0.299, kg=0.587, kb=0.114, luma_factor=1, chroma_factor=1):
        self.kr, self.kg, self.kb = kr, kg, kb
        self.luma_factor = luma_factor
        self.chroma_factor = chroma_factor

    def from_rgb(self, imgs):
        r, g, b = imgs.permute(1, 0, 2, 3)

        y = r * self.kr + g * self.kg + b * self.kb
        pb = (b - y) / (2 * (1 - self.kb))
        pr = (r - y) / (2 * (1 - self.kr))

        return torch.stack(
            [
                y * self.luma_factor,
                pb * self.chroma_factor + 0.5,
                pr * self.chroma_factor + 0.5,
            ],
            1,
        )

    def to_rgb(self, imgs):
        y_prime, pb_prime, pr_prime = imgs.permute(1, 0, 2, 3)
        y = y_prime / self.luma_factor
        pb = (pb_prime - 0.5) / self.chroma_factor
        pr = (pr_prime - 0.5) / self.chroma_factor

        b = pb * 2 * (1 - self.kb) + y
        r = pr * 2 * (1 - self.kr) + y
        g = (y - r * self.kr - b * self.kb) / self.kg

        return torch.stack([r, g, b], 1).clamp(0, 1)
