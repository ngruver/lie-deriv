import torch
from .transforms import *

def jvp(f, x, u):
    """Jacobian vector product Df(x)u vs typical autograd VJP vTDF(x).
    Uses two backwards passes: computes (vTDF(x))u and then derivative wrt to v to get DF(x)u"""
    with torch.enable_grad():
        y = f(x)
        v = torch.ones_like(
            y, requires_grad=True
        )  # Dummy variable (could take any value)
        vJ = torch.autograd.grad(y, [x], [v], create_graph=True)
        Ju = torch.autograd.grad(vJ, [v], [u], create_graph=True)
        return Ju[0]


def translation_lie_deriv(model, inp_imgs, axis="x"):
    """Lie derivative of model with respect to translation vector, output can be a scalar or an image"""
    # vector = vector.to(inp_imgs.device)
    if not img_like(inp_imgs.shape):
        return 0.0

    def shifted_model(t):
        # print("Input shape",inp_imgs.shape)
        shifted_img = translate(inp_imgs, t, axis)
        z = model(shifted_img)
        # print("Output shape",z.shape)
        # if model produces an output image, shift it back
        if img_like(z.shape):
            z = translate(z, -t, axis)
        # print('zshape',z.shape)
        return z

    t = torch.zeros(1, requires_grad=True, device=inp_imgs.device)
    lie_deriv = jvp(shifted_model, t, torch.ones_like(t, requires_grad=True))
    # print('Liederiv shape',lie_deriv.shape)
    # print(model.__class__.__name__)
    # print('')
    return lie_deriv


def rotation_lie_deriv(model, inp_imgs):
    """Lie derivative of model with respect to rotation, assumes scalar output"""
    if not img_like(inp_imgs.shape):
        return 0.0

    def rotated_model(t):
        rotated_img = rotate(inp_imgs, t)
        z = model(rotated_img)
        if img_like(z.shape):
            z = rotate(z, -t)
        return z

    t = torch.zeros(1, requires_grad=True, device=inp_imgs.device)
    lie_deriv = jvp(rotated_model, t, torch.ones_like(t))
    return lie_deriv


def hyperbolic_rotation_lie_deriv(model, inp_imgs):
    """Lie derivative of model with respect to rotation, assumes scalar output"""
    if not img_like(inp_imgs.shape):
        return 0.0

    def rotated_model(t):
        rotated_img = hyperbolic_rotate(inp_imgs, t)
        z = model(rotated_img)
        if img_like(z.shape):
            z = hyperbolic_rotate(z, -t)
        return z

    t = torch.zeros(1, requires_grad=True, device=inp_imgs.device)
    lie_deriv = jvp(rotated_model, t, torch.ones_like(t))
    return lie_deriv


def scale_lie_deriv(model, inp_imgs):
    """Lie derivative of model with respect to rotation, assumes scalar output"""
    if not img_like(inp_imgs.shape):
        return 0.0

    def scaled_model(t):
        scaled_img = scale(inp_imgs, t)
        z = model(scaled_img)
        if img_like(z.shape):
            z = scale(z, -t)
        return z

    t = torch.zeros(1, requires_grad=True, device=inp_imgs.device)
    lie_deriv = jvp(scaled_model, t, torch.ones_like(t))
    return lie_deriv


def shear_lie_deriv(model, inp_imgs, axis="x"):
    """Lie derivative of model with respect to shear, assumes scalar output"""
    if not img_like(inp_imgs.shape):
        return 0.0

    def sheared_model(t):
        sheared_img = shear(inp_imgs, t, axis)
        z = model(sheared_img)
        if img_like(z.shape):
            z = shear(z, -t, axis)
        return z

    t = torch.zeros(1, requires_grad=True, device=inp_imgs.device)
    lie_deriv = jvp(sheared_model, t, torch.ones_like(t))
    return lie_deriv


def stretch_lie_deriv(model, inp_imgs, axis="x"):
    """Lie derivative of model with respect to stretch, assumes scalar output"""
    if not img_like(inp_imgs.shape):
        return 0.0

    def stretched_model(t):
        stretched_img = stretch(inp_imgs, t, axis)
        z = model(stretched_img)
        if img_like(z.shape):
            z = stretch(z, -t, axis)
        return z

    t = torch.zeros(1, requires_grad=True, device=inp_imgs.device)
    lie_deriv = jvp(stretched_model, t, torch.ones_like(t))
    return lie_deriv


def saturate_lie_deriv(model, inp_imgs):
    """Lie derivative of model with respect to saturation, assumes scalar output"""
    if not img_like(inp_imgs.shape):
        return 0.0

    def saturated_model(t):
        saturated_img = saturate(inp_imgs, t)
        z = model(saturated_img)
        if img_like(z.shape):
            z = saturate(z, -t)
        return z

    t = torch.zeros(1, requires_grad=True, device=inp_imgs.device)
    lie_deriv = jvp(saturated_model, t, torch.ones_like(t))
    return lie_deriv

