import torch
import torch.nn.functional as F
from torch_geometric.nn import HeteroConv, Linear, MLP, SAGEConv
from torch_scatter import scatter_sum,scatter_max

class HeteroGNN(torch.nn.Module):
    def __init__(self, metadata,edge_dim,hidden_channels, out_channels, num_layers):
        super().__init__()
        self.edge_mlp = MLP(channel_list=[128+8,512,64,16],dropout=0.1)
        self.lin_mpl = Linear(in_channels=16,out_channels=16)
        self.edge_lin = Linear(in_channels=1,out_channels=8) 

        self.conv_1 = HeteroConv(
            {
                edge_type: SAGEConv([-1,-1],hidden_channels)
                            for edge_type in metadata[1]
            }
        )
        self.conv_2 = HeteroConv(
            {
                edge_type: SAGEConv([-1,-1],hidden_channels)
                            for edge_type in metadata[1]
            }
        )
        self.conv_3 = HeteroConv(
            {
                edge_type: SAGEConv([-1,-1],hidden_channels)
                            for edge_type in metadata[1]
            }
        )


    def forward(self, x_dict, edge_index_dict,edge_attr_dict,batch_dict):

        x1_dict = self.conv_1(x_dict, edge_index_dict)
        x1_dict = {key: F.leaky_relu(x) for key, x in x1_dict.items()}

        x2_dict = self.conv_2(x1_dict, edge_index_dict)
        x2_dict = {key: F.leaky_relu(x) for key, x in x2_dict.items()}

        x3_dict = self.conv_3(x2_dict, edge_index_dict)
        x3_dict = {key: F.leaky_relu(x) for key, x in x3_dict.items()}

        x_dict['ligand'] = x1_dict['ligand'] + x2_dict['ligand'] + x3_dict['ligand']
        x_dict['protein'] = x1_dict['protein'] + x2_dict['protein'] + x3_dict['protein']

        src, dst = edge_index_dict[('ligand','to','protein')]
        edge_repr = torch.cat([x_dict['ligand'][src], x_dict['protein'][dst]], dim=-1)

        d_pl = self.edge_lin(edge_attr_dict[('ligand','to','protein')])
        edge_repr = torch.cat((edge_repr,d_pl),dim=1)
        m_pl = self.edge_mlp(edge_repr)
        edge_batch = batch_dict['ligand'][src]

        w_pl = torch.tanh(self.lin_mpl(m_pl))
        m_w =  w_pl * m_pl
        m_w = scatter_sum(m_w, edge_batch, dim=0)

        m_max,_ = scatter_max(m_pl,edge_batch,dim=0)
        m_out = torch.cat((m_w, m_max), dim=1)

        return m_out
