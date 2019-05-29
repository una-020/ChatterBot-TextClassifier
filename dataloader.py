from torch.utils.data import Dataset
from utils import *


class TextDataset(Dataset):
    def __init__(self, X, Y):
        self.X = X
        self.Y = Y
        self.max_len = get_max_len(X)

    def __getitem__(self, index_model):
        index, model = index_model
        sentence = self.X[index]
        feature = model.getX([sentence])
        feature = feature.squeeze(axis=0)
        feature = torch.from_numpy(feature)
        return (feature, self.Y[index])

    def __len__(self):
        return self.X.shape[0]
