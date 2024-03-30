import torch
import torch.nn as nn
from dmasif_encoder.protein_surface_encoder import dMaSIF
from dmasif_encoder.data_iteration import iterate
from gvp.models import GVPModel
from torch_geometric.utils import to_dense_batch


class ProteinMultimodalNetwork(nn.Module):
    def __init__(self, config, device):
        super(ProteinMultimodalNetwork, self).__init__()

        self.device = device
        self.protein_seq_mpl = nn.Linear(1024, 128)

        self.protein_model = GVPModel(node_in_dim = (6,3), node_h_dim = (128, 32), edge_in_dim = (32, 1), edge_h_dim=(32, 1), 
                                num_layers = 3, drop_rate=0.3)

        self.protein_surface_encoder = dMaSIF(config.model.dmasif)

        self.sur_layers = nn.ModuleList(
            [
                nn.TransformerEncoderLayer(128, 8, 128*2, batch_first=True,dropout=0.3)
                for _ in range(1)
            ]
        )
        self.struc_layers = nn.ModuleList(
            [
                nn.TransformerDecoderLayer(128, 8, 128*2, batch_first=True,dropout=0.3)
                for _ in range(1)
            ]
        )
        self.seq_layers = nn.ModuleList(
            [
                nn.TransformerDecoderLayer(128, 8, 128*2, batch_first=True,dropout=0.3)
                for _ in range(1)
            ]
        )
              
    def forward(self, pro_seq, protein, protein_graph, ligand_center):
        p_seq = self.protein_seq_mpl(pro_seq.seq)
        p_seq, seq_mask = to_dense_batch(p_seq, pro_seq.seq_batch, max_num_nodes=512)

        
        p_res = self.protein_model((protein_graph.node_s, protein_graph.node_v), 
                                    protein_graph.edge_index, (protein_graph.edge_s, protein_graph.edge_v), protein_graph.batch)

        p_sur = iterate(self.protein_surface_encoder,ligand_center,protein)

        p_struc, p_struc_mask = to_dense_batch(p_res, protein_graph.batch)

        for i, layer in enumerate(self.sur_layers):
            p_sur_new = layer(p_sur)
        
        for i, layer in enumerate(self.struc_layers):
            p_struc_new = layer(p_struc, p_sur_new, tgt_key_padding_mask=~p_struc_mask)

        for i, layer in enumerate(self.seq_layers):
            p_seq_new = layer(p_seq, p_sur_new, tgt_key_padding_mask=~seq_mask)
           

        p_sur_new = p_sur_new.mean(dim=1)
        p_struc_new = p_struc_new.mean(dim=1)
        p_seq_new = p_seq_new.mean(dim=1)

        prot = torch.cat([p_seq_new, p_struc_new, p_sur_new], dim = -1)

        return prot
    