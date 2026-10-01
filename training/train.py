import torch
from tqdm import tqdm


def train(model, train_data, val_data=None, n_epochs=200, lr=1e-3, device="cuda",
          grad_clip=1.0):
    """
    Train the unrolled model. The loss is the KKT residual at the last
    unrolled iteration.

    train_data, val_data : lists of (initial_state, clean, functions)

    Returns (model, train_loss_history, val_loss_history).
    All metrics are shown in the progress bars.
    """
    model = model.to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-5)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=n_epochs, eta_min=1e-5)

    train_loss_hist, val_loss_hist = [], []
    epoch_bar = tqdm(range(n_epochs), desc="Training")

    for epoch in epoch_bar:

        model.train()
        losses = []
        batch_bar = tqdm(train_data, desc=f"Epoch {epoch + 1}/{n_epochs}", leave=False)

        for _, _, functions in batch_bar:
            optimizer.zero_grad()
            loss = model(functions)[0][-1]

            if not torch.isfinite(loss):
                tqdm.write(f"[Warning] non-finite loss at epoch {epoch}, skipping batch")
                continue

            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
            optimizer.step()

            losses.append(loss.item())
            batch_bar.set_postfix(loss=f"{losses[-1]:.4f}")

        scheduler.step()
        train_loss_hist.append(sum(losses) / len(losses) if losses else float("nan"))
        metrics = {"train": f"{train_loss_hist[-1]:.4f}"}

        if val_data is not None:
            model.eval()
            with torch.no_grad():
                val = [model(functions)[0][-1].item() for _, _, functions in val_data]
            val_loss_hist.append(sum(val) / len(val))
            metrics["val"] = f"{val_loss_hist[-1]:.4f}"

        metrics["lr"] = f"{scheduler.get_last_lr()[0]:.1e}"
        epoch_bar.set_postfix(metrics)

    return model, train_loss_hist, val_loss_hist
