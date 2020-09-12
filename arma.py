# -*- coding: utf-8 -*-
"""
Graph classification using ARMA filter
    
@article{bianchi2019graph,
  title={Graph neural networks with convolutional arma filters},
  author={Bianchi, Filippo Maria and Grattarola, Daniele and Alippi, Cesare and Livi, Lorenzo},
  journal={arXiv preprint arXiv:1901.01343},
  year={2019}
}


"""
import torch
import torch.nn.functional as F
from torch import nn
from torch_geometric.datasets import TUDataset
from torch_geometric.data import DataLoader
from torch_geometric.nn import ARMAConv, global_mean_pool
from torch_geometric.utils import to_networkx
import networkx as nx

##--
class add_nodeFeatures(object):
    r"""Add
    """

    def __call__(self, Data):

        g = to_networkx(Data,to_undirected=True)

        deg, cluster_coeff = torch.FloatTensor(list(dict(nx.degree(g)).values())), torch.FloatTensor(list(nx.clustering(g).values()))

        x = Data.x
        if x is not None:
            x = x.view(-1, 1) if x.dim() == 1 else x
            Data.x = torch.cat([Data.x, deg.view(Data.num_nodes,1), cluster_coeff.view(Data.num_nodes,1)], dim=1)

        return Data
##--  

##--
class Net(torch.nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        
        #num_stacks stands for K order polynomial/rational filter
        #num_layer stack depths
        self.conv1 = ARMAConv(dataset.num_features, 32, num_stacks=2, shared_weights=True,
                              num_layers=2, dropout=0.6)
        
        self.conv2 = ARMAConv(32, 32, num_stacks=2, shared_weights=True,
                               num_layers=2, dropout=0.6)
        self.conv3 = ARMAConv(32, 32, num_stacks=2, shared_weights=True,
                              num_layers=2, dropout=0.6)
        
        self.FullyConnected = nn.Linear(32, dataset.num_classes)
        
    #Batch per il val?
    def forward(self, data):
        x, edge_index, batch = data.x, data.edge_index, data.batch

        x = F.relu(self.conv1(x, edge_index))

        x = F.relu(self.conv2(x, edge_index))
        
        x = F.relu(self.conv3(x,edge_index))
        
        x = global_mean_pool(x, batch)
        
        x = self.FullyConnected(x)
        
        return F.log_softmax(x)
##--
##########


dataset = TUDataset(root='/tmp/ENZYMES', name='ENZYMES',use_node_attr=True, pre_transform=add_nodeFeatures()).shuffle()    


device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
train_loader = DataLoader(dataset[:(len(dataset)*80)//100 ], batch_size=32, shuffle=True)
val_loader = DataLoader(dataset[(len(dataset)*80)//100:(len(dataset)*80)//100 +  (len(dataset)*10)//100 ], batch_size=32, shuffle=True)
model= Net().to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=1e-3, weight_decay=1e-4)


def train():
    for batch in train_loader:
        batch.to(device)
        model.train()
        optimizer.zero_grad()
        loss = F.nll_loss(model(batch), batch.y)
        loss.backward()
        optimizer.step()
        return loss.item()

def test():
    pred = torch.empty(0,device= device)
    for batch in val_loader:
        batch.to(device)
        model.eval()
        _, yhat = torch.max(model(batch), 1)
        pred = torch.cat((pred,(yhat == batch.y)),0)        
    return pred.mean()


loss =[]
accs = []
for epoch in range(1, 1000):
    loss_epoch = train()
    loss.append(loss_epoch)
    accs_epoch = test()
    accs.append(accs_epoch)
