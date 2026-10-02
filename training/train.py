import random

import torch
from tqdm import tqdm


def final_loss(model, clean, functions, loss):
    """Loss at the last unrolled iterate x_T = (u_T, w_T, p_T, q_T).

    "kkt"       : KKT residual ||(A + C)(x_T)||
    "objective" : TGV2 objective F(u_T, w_T), the quantity plotted in the paper
    "image"     : squared error between u_T and the ground-truth image
    """
    kkt, _, x_hist = model(functions)
    if loss == "kkt":
        return kkt[-1]
    if loss == "objective":
        return functions["objective"](x_hist[-1])
    if loss == "image":
        return (x_hist[-1][0] - clean).pow(2).sum()
    raise ValueError(f"unknown loss {loss!r}")


def train(model, train_data, val_data=None, n_epochs=200, lr=1e-3, device="cuda",
          grad_clip=1.0, ckpt_path=None, loss="kkt"):
    """
    Train the unrolled model on a loss at the last unrolled iterate
    (see final_loss).

    train_data, val_data : lists of (initial_state, clean, functions)

    After every epoch the loss histories and the weights of the best epoch
    (lowest validation loss) are saved to ckpt_path, so a long run can be
    interrupted without losing everything. The returned model has the weights
    of the best epoch.

    Returns (model, train_loss_history, val_loss_history).
    All metrics are shown in the progress bars.
    """
    def copy_weights():
        return {k: v.detach().clone() for k, v in model.state_dict().items()}

    model = model.to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-5)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=n_epochs, eta_min=1e-5)

    train_loss_hist, val_loss_hist = [], []
    best_loss, best_weights = float("inf"), copy_weights()
    epoch_bar = tqdm(range(n_epochs), desc="Training")

    for epoch in epoch_bar:

        model.train()
        losses = []
        shuffled = random.sample(train_data, len(train_data))
        batch_bar = tqdm(shuffled, desc=f"Epoch {epoch + 1}/{n_epochs}", leave=False)

        for _, clean, functions in batch_bar:
            optimizer.zero_grad()
            value = final_loss(model, clean, functions, loss)

            if not torch.isfinite(value):
                tqdm.write(f"[Warning] non-finite loss at epoch {epoch}, skipping batch")
                continue

            value.backward()
            grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
            # a single non-finite gradient would turn every weight into NaN
            if not torch.isfinite(grad_norm):
                tqdm.write(f"[Warning] non-finite gradient at epoch {epoch}, skipping batch")
                continue
            optimizer.step()

            losses.append(value.item())
            batch_bar.set_postfix(loss=f"{losses[-1]:.4f}")

        scheduler.step()
        train_loss_hist.append(sum(losses) / len(losses) if losses else float("nan"))
        metrics = {"train": f"{train_loss_hist[-1]:.4f}"}

        if val_data is not None:
            model.eval()
            with torch.no_grad():
                val = [final_loss(model, clean, functions, loss).item()
                       for _, clean, functions in val_data]
            val_loss_hist.append(sum(val) / len(val))
            metrics["val"] = f"{val_loss_hist[-1]:.4f}"

        # keep the best epoch (validation loss if available, else training loss)
        current = val_loss_hist[-1] if val_data is not None else train_loss_hist[-1]
        if current < best_loss:
            best_loss, best_weights = current, copy_weights()
        if ckpt_path is not None:
            torch.save({"model": best_weights, "train_loss_history": train_loss_hist,
                        "val_loss_history": val_loss_hist, "best_loss": best_loss,
                        "primal_step": model.params.primal_step, "loss": loss}, ckpt_path)

        metrics["best"] = f"{best_loss:.4f}"
        metrics["lr"] = f"{scheduler.get_last_lr()[0]:.1e}"
        epoch_bar.set_postfix(metrics)

    model.load_state_dict(best_weights)
    return model, train_loss_hist, val_loss_hist
