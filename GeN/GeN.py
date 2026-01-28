import torch
import numpy as np

from scipy.optimize import curve_fit
from sklearn.metrics import r2_score

def fit_func2(x, gHg, Gg):
    return 0.5*gHg * x**2 - Gg * x

def setup(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    np.random.seed(seed)
    # random.seed(seed)
    torch.backends.cudnn.benchmark = True
    torch.backends.cudnn.deterministic = True


def lr_parabola(
    net, 
    optimizer, 
    criterion=torch.nn.CrossEntropyLoss(), 
    grad_accumulation_steps=1,
    tr_iter=None, 
    device='cuda',
    task=None,
    scale=1.0,
    lr_factor=[-1,0,1],
    verbose=False
):
    """
    Adaptively updates the optimizer's learning rate using a parabolic fit to the loss landscape.
    This function implements the Generalized Newton's method (GeN) for learning-rate-free optimization.

    Args:
        net (torch.nn.Module): The model to be optimized.
        optimizer (torch.optim.Optimizer): The optimizer that pre-conditions the gradient.
        criterion (callable, optional): Loss function. Default is torch.nn.CrossEntropyLoss().
        grad_accumulation_steps (int, optional): Number of gradient accumulation steps. Default is 1.
        tr_iter (iterator, optional): Training data iterator for multiple forward passes on the same batch.
        device (str, optional): Device to use ('cuda' or 'cpu'). Default is 'cuda'.
        task (str, optional): Task type, one of 'image_cls', 'NLG', 'NLU', 'time_series'. Customized task can be added by wrapping the forward pass.
        scale (float, optional): Dependence on training horizon. Default is 1.0 (no dependence). Alternatively, set to a hyper-parameter-free decay from 1 to 0, e.g. 1-t/T.
        lr_factor (list, optional): List of candidate learning rate multipliers. Default is [-1,0,1].
        verbose(bool, optional): Whether to print learning rates and losses. Default is False.

    Usage:
        >>> for batch in train_loader:
        ...     loss = criterion(model(batch), labels)
        ...     loss.backward()
        ...     if (step+1) % lazy_freq == 0:
        ...         lr_parabola(model, optimizer, tr_iter=tr_iter, task='image_cls', scale=scale)
        ...     optimizer.step()
        ...     optimizer.zero_grad()

    Notes:
        - Call infrequently (lazy update) for efficiency.
        - Supports multiple tasks by switching the forward function based on 'task'.
        - Uses in-place operations to save memory.
    """

    assert lr_factor.count(0) == 1, "lr_factor must contain exactly one 0"
    lr_factor.append(lr_factor.pop(lr_factor.index(0))) # move 0 to the end

    existing_lr = optimizer.param_groups[0]['lr']
    for param in net.parameters():
        if param.requires_grad:
            param.previous = param.data.clone() # copy w_t
    optimizer.step()

    with torch.no_grad():
        lr_ratio_list = np.array(lr_factor, dtype='float64')
        loss_list = np.zeros_like(lr_ratio_list)

        for _ in range(grad_accumulation_steps):
            next_data = next(tr_iter)

            if len(next_data) == 2 and task == 'image_cls':
                inputs = next_data[0].to(device)
                targets = next_data[1].to(device)
            if len(next_data) == 2 and task == 'time_series':
                inputs = next_data[0].to(device)
                targets = next_data[1].to(device)
            if len(next_data) > 2 and task == 'NLG':
                # next_data = {key: value for key, value in next_data.items()}
                inputs = next_data['input'].to(device)
                targets = next_data['target'].to(device)
                masks = next_data['mask'].to(device)
            if task == 'NLU':
                batch = next_data.to(device)

            for j, ratio in enumerate(lr_ratio_list):
                for param in net.parameters():
                    if param.requires_grad:
                        param.data.mul_(ratio).add_(param.previous, alpha=(1.0 - ratio))
                        # in-place operation to create param.data <-- w_t-ratio*lr*g

                if task == 'image_cls':
                    outputs = net(inputs)
                    loss_temp = criterion(outputs, targets) # need to return torch.tensor
                elif task == 'time_series':
                    outputs = net(inputs)
                    loss_temp = criterion(outputs, targets) # need to return torch.tensor
                elif task == 'NLG':
                    _lm_logits, loss_temp = net(inputs, lm_labels=targets, lm_mask=masks)
                elif task == 'NLU':
                    outputs = net(**batch)
                    loss_temp = outputs.loss

                loss_list[j] += loss_temp.item()

                for param in net.parameters():
                    if param.requires_grad and ratio != 0:
                        param.data.sub_(param.previous, alpha=(1.0 - ratio)).div_(ratio)
                        # in-place operation to recover param.data <-- w_t

        try:
            smooth = 0.9
            lr_list = lr_ratio_list * existing_lr
            [gHg, Gg] = curve_fit(fit_func2, lr_list, loss_list - loss_list[-1])[0]
            r2_current = r2_score(loss_list - loss_list[-1], fit_func2(lr_list, gHg, Gg))

            if gHg > 0 and Gg > 0 and r2_current > 0.99:
                opt_lr = Gg / gHg
                for g in optimizer.param_groups:
                    g['lr'] = (max(min(opt_lr, g['lr'] * 2), g['lr'] / 2) * (1 - smooth) * scale + smooth * g['lr'])
                if verbose:
                    print(f'lr={lr_list}, loss={loss_list}, R2 score={r2_current}')
                    print(f">>> current lr={optimizer.param_groups[0]['lr']}")
            for param in net.parameters():
                if param.requires_grad:
                    del param.previous
        except Exception as e:
            for g in optimizer.param_groups:
                g['lr'] = g['lr'] / 2
            print(e)
