"""export_gfpgan.py - GFPGAN v1.4 (the .pth) as an ONNX model, for the "Sharper face" switch. Run once, with a Python that
has gfpgan installed (Changeling Imago's psdenv does):

    ..\Changeling-Imago\psdenv\Scripts\python.exe scripts\export_gfpgan.py  path\to\GFPGANv1.4.pth  models\gfpgan_1.4.onnx

In: 1 x 3 x 512 x 512 RGB in -1..1 (a face aligned as FFHQ is); out: the same, the face redrawn sharp.
"""
import sys

import torch
try:
    import torchvision.transforms.functional_tensor  # noqa: F401  (basicsr wants it; newer torchvision moved it)
except ImportError:
    import torchvision.transforms.functional
    sys.modules["torchvision.transforms.functional_tensor"] = torchvision.transforms.functional
from gfpgan.archs.gfpganv1_clean_arch import GFPGANv1Clean


class Only(torch.nn.Module):
    def __init__(self, net):
        super().__init__()
        self.net = net

    def forward(self, x):
        return self.net(x, return_rgb=False, randomize_noise=False)[0]


def main(src, dst):
    net = GFPGANv1Clean(out_size=512, num_style_feat=512, channel_multiplier=2, decoder_load_path=None,
                        fix_decoder=False, num_mlp=8, input_is_latent=True, different_w=True, narrow=1, sft_half=True)
    state = torch.load(src, map_location="cpu")
    net.load_state_dict(state.get("params_ema", state.get("params", state)), strict=True)
    net.eval()
    torch.onnx.export(Only(net), torch.zeros(1, 3, 512, 512), dst, input_names=["input"], output_names=["output"],
                      opset_version=17, do_constant_folding=True)
    print(f"Saved {dst}")


if __name__ == "__main__":
    main(*sys.argv[1:3])
