import torch


def trajectory_loss(kkt_list, gamma=0.85, log_domain=True, eps=1e-12):

    T = len(kkt_list)
    weights = [gamma ** (T - 1 - t) for t in range(T)]
    norm = sum(weights)

    def term(k):
        k = k.mean()
        return torch.log(k + eps) if log_domain else k

    loss = sum(w * term(k) for w, k in zip(weights, kkt_list)) / norm
    return loss
