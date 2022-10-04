from email import message_from_bytes
from mailbox import Message
import torch
import torch.nn.functional as F
import pandas as pd
import numpy as np

import sys
sys.path.append("../stylegan3")
from stylegan3.metrics.equivariance import (
    apply_integer_translation,
    apply_fractional_translation,
    apply_fractional_rotation,
)

from transforms import (
    translate,
    rotate,
)

def EQ_T(model, img, model_out, translate_max=0.125):
    d = []
    for _ in range(10):
        t = (torch.rand(2, device='cuda') * 2 - 1) * translate_max
        t = (t * img.shape[-2] * img.shape[-1]).round() / (img.shape[-2] * img.shape[-1])
        ref, _ = apply_integer_translation(img, t[0], t[1])
        t_model_out = model(ref)
        d.append((t_model_out - model_out).square().mean())
    #psnr = np.log10(2) * 20 - mse.log10() * 10
    return torch.stack(d).mean()

def EQ_T_frac(model, img, model_out, translate_max=0.125):
    d = []
    for _ in range(10):
        t = (torch.rand(2, device='cuda') * 2 - 1) * translate_max
        ref, _ = apply_fractional_translation(img, t[0], t[1])
        t_model_out = model(ref)
        d.append((t_model_out - model_out).square().mean())
    # psnr = np.log10(2) * 20 - mse.log10() * 10
    return torch.stack(d).mean()

def EQ_R(model, img, model_out, rotate_max=1.0):
    d = []
    for _ in range(10):
        angle = (torch.rand([], device='cuda') * 2 - 1) * (rotate_max * np.pi)
        ref, _ = apply_fractional_rotation(img, angle)
        t_model_out = model(ref)
        d.append((t_model_out - model_out).square().mean())
    # psnr = np.log10(2) * 20 - mse.log10() * 10
    return torch.stack(d).mean()


def translation_sample_invariance(model,inp_imgs,model_out,axis='x',eta=2.0):
    """ Lie derivative of model with respect to translation vector, assumes scalar output """
    shifted_model = lambda t: model(translate(inp_imgs,t,axis))
    d = []
    for _ in range(10):
        t_sample = (2 * eta) * torch.rand(1) - eta
        d.append((model_out - shifted_model(t_sample)).pow(2).mean())
    return torch.stack(d).mean(0).unsqueeze(0)

def rotation_sample_invariance(model,inp_imgs,model_out,eta=np.pi//16):
    """ Lie derivative of model with respect to rotation, assumes scalar output """
    rotated_model = lambda theta: model(rotate(inp_imgs,theta))
    d = []
    for _ in range(10):
        theta_sample = (2 * eta) * torch.rand(1) - eta
        d.append((model_out - rotated_model(theta_sample)).pow(2).mean())
    return torch.stack(d).mean(0).unsqueeze(0)


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

    with torch.no_grad():
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

        eq_t = torch.stack([EQ_T(model_probs, x, model_out) for _ in range(num_probes)], dim=0).mean(0)
        eq_t_frac = torch.stack([EQ_T_frac(model_probs, x, model_out) for _ in range(num_probes)], dim=0).mean(0)
        eq_r = torch.stack([EQ_R(model_probs, x, model_out) for _ in range(num_probes)], dim=0).mean(0)

        metrics["eq_t"] = eq_t.cpu().data.numpy()
        metrics["eq_t_frac"] = eq_t_frac.cpu().data.numpy()
        metrics["eq_r"] = eq_r.cpu().data.numpy()
    
        metrics['trans_x_sample'] = translation_sample_invariance(model_probs,x,model_out,axis='x').abs().cpu().data.numpy()
        metrics['trans_y_sample'] = translation_sample_invariance(model_probs,x,model_out,axis='y').abs().cpu().data.numpy()
        metrics['rotate_sample'] = rotation_sample_invariance(model_probs,x,model_out).abs().cpu().data.numpy()

    df = pd.DataFrame.from_dict(metrics)
    return df
