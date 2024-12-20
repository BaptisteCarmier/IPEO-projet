import torch


def compute_mae(predictions, masks):
    # Flatten the tensors to compute pixel-wise error
    predictions = predictions.view(-1)
    masks = masks.view(-1)

    # Compute the mean absolute error
    mae = torch.mean(torch.abs(predictions - masks))
    return mae

def compute_mse(predictions, masks):
    # Flatten the tensors to compute pixel-wise error
    predictions = predictions.view(-1)
    masks = masks.view(-1)

    # Compute the mean squared error
    mse = torch.mean((predictions - masks) ** 2)
    return mse

def compute_r2(predictions, masks):
    # Flatten the tensors
    predictions = predictions.view(-1)
    masks = masks.view(-1)

    # Compute the total sum of squares and residual sum of squares
    ss_total = torch.sum((masks - torch.mean(masks)) ** 2)
    ss_residual = torch.sum((masks - predictions) ** 2)

    # Compute R-squared
    r2 = 1 - ss_residual / ss_total
    return r2

def compute_mape(predictions, masks):
    # Flatten the tensors
    predictions = predictions.view(-1)
    masks = masks.view(-1)

    # Avoid division by zero
    epsilon = 1e-6
    mape = torch.mean(torch.abs((predictions - masks) / (masks + epsilon))) * 100
    return mape
