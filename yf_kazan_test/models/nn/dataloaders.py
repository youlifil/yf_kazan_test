from torch.utils.data import DataLoader, TensorDataset
from sklearn.model_selection import train_test_split

def nn_dataloaders(datapack, batch_size=64):
    train_val_dataset = TensorDataset(datapack.X_train, datapack.y_train)
    train_dataset, val_dataset = train_test_split(train_val_dataset, test_size=0.2, random_state=47)
    test_dataset = TensorDataset(datapack.X_test, datapack.y_test)

    train_dl = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_dl = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
    test_dl = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    return {
        "train": train_dl,
        "val": val_dl,
        "test": test_dl
    }
