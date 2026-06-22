import torch


def cal_metrics(output, target, var_name):
    output = output.detach()
    target = target.detach()
    metrics = {}

    for idx, name in enumerate(var_name):
        out = output[:, :, idx]
        tgt = target[:, :, idx]
        rmse = torch.sqrt(((out - tgt) ** 2).mean()).float()

        metrics[name] = rmse

    metrics['score'] = torch.stack(list(metrics.values())).mean()

    return metrics
