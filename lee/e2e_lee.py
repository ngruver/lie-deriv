import torch
import pandas as pd
from .lie_derivs import *

def get_equivariance_metrics(model, minibatch):
    x, y = minibatch
    if torch.cuda.is_available():
        model = model.cuda()
        x, y = x.cuda(), y.cuda()

    model = model.eval()

    model_probs = lambda x: F.softmax(model(x), dim=-1)

    errs = {
        "trans_x_deriv": translation_lie_deriv(model_probs, x, axis="x"),
        "trans_y_deriv": translation_lie_deriv(model_probs, x, axis="y"),
        "rot_deriv": rotation_lie_deriv(model_probs, x),
        "shear_x_deriv": shear_lie_deriv(model_probs, x, axis="x"),
        "shear_y_deriv": shear_lie_deriv(model_probs, x, axis="y"),
        "stretch_x_deriv": stretch_lie_deriv(model_probs, x, axis="x"),
        "stretch_y_deriv": stretch_lie_deriv(model_probs, x, axis="y"),
        "saturate_err": saturate_lie_deriv(model_probs, x),
    }
    
    metrics = {x: pd.Series(errs[x].abs().cpu().data.numpy().mean(-1)) for x in errs}
    df = pd.DataFrame.from_dict(metrics)
    return df
