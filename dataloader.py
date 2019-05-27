from torch.utils.data import Dataset


class TextDataset(Dataset):
    def __init__(self, X, Y):
        self.X = X
        self.Y = Y

    def __getitem__(self, index):
        return (self.X[index, :], self.Y[index])

    def __len__(self):
        return self.X.shape[0]
