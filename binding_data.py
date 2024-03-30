import pickle
from torch.utils.data import Dataset

class PLBA_Dataset(Dataset):
    def __init__(self, *args):
        if(args[0]=="file"):
            filepath = args[1]
            f = open(filepath, 'rb')
            self.G_list = pickle.load(f)
            self.len = len(self.G_list)
        elif(args[0]=='list'):
            self.G_list = args[1]
            self.len = len(args[1])


    def __getitem__(self, index):
        G = self.G_list[index]
        return G[0], G[1], G[2], G[4], G[5], G[6]

    def __len__(self):
        return self.len

    def k_fold(self,train_idx,val_idx):
        train_list = [ self.G_list[i] for i in train_idx ]
        val_list = [self.G_list[i] for i in val_idx ]
        return train_list,val_list

    def merge(self,data):
        self.G_list += data
        return self.G_list
    
    def len(self):
        return self.len

    def get(self, idx):
        data = self.G_list[idx]
        return data
