import torch
import torch.nn as nn

from .config import PAD_IDX, MAX_EPOCHS
from yf_kazan_test.vectors.indices import decode_category
from yf_kazan_test.models import print_score

import numpy as np

def epoch_printer(epoch, train_loss, val_loss):
    print('\rEpoch {:02d}, Training Loss: {:.6f}, Validation Loss: {:.6f}'.format(epoch+1, train_loss, val_loss))


def batch_printer(mode, epoch, batch_i, batch_total):
    print(f"\rEpoch {epoch+1:02d}: {mode} batch {batch_i:04d}/{batch_total:04d}", end="")


def train_model(model, optimizer, dataloaders, patience=None, epoch_cb=epoch_printer, batch_cb=batch_printer):
    device = "cuda" if torch.cuda.is_available() else "cpu"

    min_loss = np.inf
    cur_patience = 0

    losses = {
        "train": [],
        "val": []
    }

    loss_func = nn.CrossEntropyLoss(ignore_index=PAD_IDX)
    model = model.to(device)

    train_dl = dataloaders["train"]
    val_dl = dataloaders["val"]

    for epoch in range(MAX_EPOCHS):
        train_loss = 0.
        model.train()
        for it, batch in enumerate(train_dl):
            optimizer.zero_grad()
            X, y = batch
            pred = model(X.to(device))
            loss = loss_func(pred, y.to(device))
            loss.backward()
            optimizer.step()
            train_loss += loss.item()

            if batch_cb:
                batch_cb("Train", epoch, it, len(train_dl))

        train_loss /= len(train_dl)

        val_loss = 0.
        model.eval()
        with torch.no_grad():
            for it, batch in enumerate(val_dl):
                X, y = batch
                pred = model(X.to(device))
                loss = loss_func(pred, y.to(device))
                val_loss += loss.item()

                if batch_cb:
                    batch_cb("Validate", epoch, it, len(val_dl))

        val_loss /= len(val_dl)

        losses["train"].append(train_loss)
        losses["val"].append(val_loss)
        
        if epoch_cb:
            epoch_cb(epoch, train_loss, val_loss)

        if val_loss < min_loss:
            min_loss = val_loss
            torch.save(model.state_dict(), "model-best.pt")
            cur_patience = 0
        elif patience:
            cur_patience += 1
            if cur_patience == patience:
                break

    torch.save(model.state_dict(), "model-last.pt")

    return losses


def score_model(model, test_dl):
    device = "cuda" if torch.cuda.is_available() else "cpu"

    model.eval()
    trues = []
    preds = []

    with torch.no_grad():
        for i, batch in enumerate(test_dl):
            X, y = batch
            pred = model(X.to(device)).cpu()
            pred = pred.argmax(dim=1)
            trues.append(y)
            preds.append(pred)

    true = torch.cat(trues)
    pred = torch.cat(preds)

    true = decode_category(true.numpy())
    pred = decode_category(pred.numpy())
    print_score(true, pred)
