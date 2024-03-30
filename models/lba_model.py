import torch
import torch.nn as nn
from torch_geometric.nn import AttentiveFP
from models.pl_model import HeteroGNN
from models.protein_multimodal import ProteinMultimodalNetwork    

class LBAPredictor(torch.nn.Module):
    def __init__(self, metadata, config, device):
        super().__init__()
        self.heterognn = HeteroGNN(metadata, edge_dim=10, hidden_channels=64, out_channels=8,num_layers=3)
        self.ligandgnn = AttentiveFP(in_channels=18,hidden_channels=64,out_channels=16,edge_dim=12,num_timesteps=3,num_layers=3,dropout=0.3)
        self.pmn =  ProteinMultimodalNetwork(config, device)

        self.out = nn.Sequential(
            nn.Linear(16+32+128+128+128, 256),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(128, 1),
            nn.ReLU()
        )
        
    def forward(self, data):
        g_l = data[0]
        g_pl = data[1]
        pro_seq = data[2]
        ligand_center = data[3]
        protein = data[4]
        protein_graph = data[5]

        l = self.ligandgnn(x=g_l.x,edge_index=g_l.edge_index,edge_attr=g_l.edge_attr,batch=g_l.batch)
        complex = self.heterognn(g_pl.x_dict, g_pl.edge_index_dict, g_pl.edge_attr_dict, g_pl.batch_dict)
        prot = self.pmn(pro_seq, protein, protein_graph,ligand_center)

        emb = torch.cat((l,complex,prot),dim=1)
        y_hat = self.out(emb)
        return torch.squeeze(y_hat)