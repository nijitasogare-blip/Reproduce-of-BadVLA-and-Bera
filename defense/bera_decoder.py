"""Train a lightweight UNet inpainter to erase trigger patches (Bera decoder)."""

import os
import sys
import time

os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import tensorflow_datasets as tfds

OUT = os.environ.get("DEC_OUT", "/data/runs/attack/bera_defense/bera_decoder.pt")


class UNet(nn.Module):
    def __init__(self):
        super().__init__()
        def block(i, o):
            return nn.Sequential(nn.Conv2d(i, o, 3, padding=1), nn.ReLU(),
                                 nn.Conv2d(o, o, 3, padding=1), nn.ReLU())
        self.e1 = block(4, 32)
        self.e2 = block(32, 64)
        self.e3 = block(64, 128)
        self.pool = nn.MaxPool2d(2)
        self.up2 = nn.ConvTranspose2d(128, 64, 2, stride=2)
        self.up3 = nn.ConvTranspose2d(64, 32, 2, stride=2)
        self.d2 = block(128, 64)
        self.d3 = block(64, 32)
        self.out = nn.Conv2d(32, 3, 1)

    def forward(self, x):
        e1 = self.e1(x)
        e2 = self.e2(self.pool(e1))
        e3 = self.e3(self.pool(e2))
        u2 = torch.cat([self.up2(e3), e2], 1)
        u3 = torch.cat([self.up3(self.d2(u2)), e1], 1)
        return self.out(self.d3(u3))


def get_frames(n=160):
    b = tfds.builder("libero_spatial_no_noops", data_dir="/data/datasets/modified_libero_rlds")
    out = []
    for ep in b.as_dataset(split="train", shuffle_files=False).take(8):
        for step in ep["steps"]:
            img = np.asarray(step["observation"]["image"])
            img = img[::1, ::1]  # keep orientation as stored
            out.append(img)
            if len(out) >= n:
                return out
    return out


def make_batch(frames, dev):
    xs, ys = [], []
    for f in frames:
        img = torch.from_numpy(f).permute(2, 0, 1).float().to(dev) / 255.0
        img = torch.nn.functional.interpolate(img[None], size=(224, 224))[0]
        mask = torch.zeros(224, 224, device=dev)
        side = 22  # white block side at 224 px (~10% of 256 image)
        r = torch.randint(0, 224 - side + 1, (1,)).item()
        c = torch.randint(0, 224 - side + 1, (1,)).item()
        # Mask the grid cells (16x16, 14px each) touched by the white block.
        cells_r = range(r // 14, (r + side - 1) // 14 + 1)
        cells_c = range(c // 14, (c + side - 1) // 14 + 1)
        for rr in cells_r:
            for cc in cells_c:
                mask[rr * 14:(rr + 1) * 14, cc * 14:(cc + 1) * 14] = 1.0
        corrupted = img.clone()
        corrupted[:, r:r + side, c:c + side] = 1.0  # pure white block, same as trigger
        xs.append(torch.cat([corrupted, mask[None]], 0))
        ys.append(img)
    return torch.stack(xs), torch.stack(ys)


def main():
    os.makedirs(os.path.dirname(OUT), exist_ok=True)
    dev = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    frames = get_frames()
    print("frames:", len(frames), flush=True)
    model = UNet().to(dev)
    opt = optim.Adam(model.parameters(), lr=1e-3)
    t0 = time.time()
    steps = int(os.environ.get("DEC_STEPS", "800"))
    for step in range(steps):
        idx = np.random.choice(len(frames), 8, replace=False)
        x, y = make_batch([frames[i] for i in idx], dev)
        opt.zero_grad()
        loss = torch.nn.functional.mse_loss(model(x), y)
        loss.backward()
        opt.step()
        if step % 100 == 0:
            print(f"step {step}: loss={loss.item():.5f}", flush=True)
    torch.save(model.state_dict(), OUT)
    print(f"decoder saved ({time.time()-t0:.0f}s)", flush=True)


if __name__ == "__main__":
    main()
