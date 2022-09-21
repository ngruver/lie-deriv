from email import message_from_bytes
from mailbox import Message
import torch
import torch.nn.functional as F
import pandas as pd
import numpy as np

import sys
sys.path.append("stylegan3")
from stylegan3.metrics.equivariance import (
    apply_integer_translation,
    apply_fractional_translation,
    apply_fractional_rotation,
)

def EQ_T(model, img, model_out, translate_max=0.125):
    t = (torch.rand(2, device='cuda') * 2 - 1) * translate_max
    t = (t * img.shape[-2] * img.shape[-2]).round() / (img.shape[-2] * img.shape[-1])
    ref, _ = apply_integer_translation(img, t[0], t[1])
    t_model_out = model(ref)
    mse = (t_model_out - model_out).square().mean()
    #psnr = np.log10(2) * 20 - mse.log10() * 10
    return mse

def EQ_T_frac(model, img, model_out, translate_max=0.125):
    t = (torch.rand(2, device='cuda') * 2 - 1) * translate_max
    ref, _ = apply_fractional_translation(img, t[0], t[1])
    t_model_out = model(ref)
    mse = (t_model_out - model_out).square().mean()
    # psnr = np.log10(2) * 20 - mse.log10() * 10
    return mse

def EQ_R(model, img, model_out, rotate_max=1.0):
    angle = (torch.rand([], device='cuda') * 2 - 1) * (rotate_max * np.pi)
    ref, _ = apply_fractional_rotation(img, angle)
    t_model_out = model(ref)
    mse = (t_model_out - model_out).square().mean()
    # psnr = np.log10(2) * 20 - mse.log10() * 10
    return mse

def get_equivariance_metrics(model, minibatch, num_probes=20):
    x, y = minibatch
    if torch.cuda.is_available():
        model = model.cuda()
        x, y = x.cuda(), y.cuda()

    model = model.eval()

    model_probs = lambda x: F.softmax(model(x), dim=-1)
    model_out = model_probs(x)

    yhat = model_out.argmax(dim=1)  # .cpu()
    acc = (yhat == y).cpu().float().data.numpy()

    metrics = {}
    metrics["acc"] = pd.Series(acc)

    for shift_x in range(8):
        rolled_img = torch.roll(x, shift_x, 2)
        rolled_yhat = model(rolled_img).argmax(dim=1)
        consistency = (rolled_yhat == yhat).cpu().data.numpy()
        metrics["consistency_x" + str(shift_x)] = pd.Series(consistency)
    for shift_y in range(8):
        rolled_img = torch.roll(x, shift_y, 3)
        rolled_yhat = model(rolled_img).argmax(dim=1)
        consistency = (rolled_yhat == yhat).cpu().data.numpy()
        metrics["consistency_y" + str(shift_y)] = pd.Series(consistency)

    eq_t = torch.stack([EQ_T(model, x, model_out) for _ in range(num_probes)], dim=0).mean(0)
    eq_t_frac = torch.stack([EQ_T_frac(model, x, model_out) for _ in range(num_probes)], dim=0).mean(0)
    eq_r = torch.stack([EQ_R(model, x, model_out) for _ in range(num_probes)], dim=0).mean(0)

    metrics["eq_t"] = eq_t.cpu().data.numpy()
    metrics["eq_t_frac"] = eq_t_frac.cpu().data.numpy()
    metrics["eq_r"] = eq_r.cpu().data.numpy()

    df = pd.DataFrame.from_dict(metrics)
    return df
