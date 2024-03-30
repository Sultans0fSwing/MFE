import os
import torch
import warnings
from sklearn import metrics
warnings.filterwarnings('ignore')
import matplotlib.pyplot as plt
import torch.nn.functional as F
from torch_geometric.data import DataLoader
from sklearn.linear_model import LinearRegression
from easydict import EasyDict
import scipy
import yaml

from models.lba_model import LBAPredictor
from binding_data import PLBA_Dataset
from dmasif_encoder.data_iteration import iterate_surface_precompute


seed = 100
torch.manual_seed(seed)
torch.cuda.manual_seed(seed)
device = torch.device('cuda')


def set_gpu(data, device):
    data_gpu = []
    for g in data:
        data_gpu.append(g.to(device))
    return data_gpu


def metrics_reg(targets,predicts):
    mae = metrics.mean_absolute_error(y_true=targets,y_pred=predicts)
    rmse = metrics.mean_squared_error(y_true=targets,y_pred=predicts,squared=False)
    r = scipy.stats.mstats.pearsonr(targets, predicts)[0]

    x = [ [item] for item in predicts]
    lr = LinearRegression()
    lr.fit(X=x,y=targets)
    y_ = lr.predict(x)
    sd = (((targets - y_) ** 2).sum() / (len(targets) - 1)) ** 0.5

    return [mae,rmse,r,sd]


def my_train(train_loader, val_loader, test_loader, kf_filepath, model, config):
    print('start training')

    optimizer = torch.optim.Adam(model.parameters(), lr=0.0001, weight_decay=5e-4)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode = "min", factor = 0.5, patience=5, verbose = True)

    loss_list = []
    best_mae = float('inf')
    best_rmse = float('inf')
    for epoch in range(50):
        model.train()
        loss_epoch = 0
        n = 0
        for data in train_loader:
            data = set_gpu(data, device)
            optimizer.zero_grad()
            out = model(data)

            loss = F.mse_loss(out, data[0].y)
            loss_epoch += loss.item()

            print('epoch:', epoch, ' i', n, ' loss:', loss.item())
            
            loss.backward()
            optimizer.step()
            n += 1
        loss_list.append(loss_epoch / n)

        print('epoch:', epoch, ' loss:', loss_epoch / n)

        val_err = my_val(model, val_loader, device, scheduler)
        val_mae = val_err[0]
        val_rmse = val_err[1]

        if val_rmse < best_rmse and val_mae < best_mae:
            print('********save model*********')
            torch.save(model.state_dict(), kf_filepath+'best_model.pt')
            best_mae = val_mae
            best_rmse = val_rmse

            f_log = open(file=(kf_filepath+"/metrics_log.txt"), mode="a")
            str_log = 'epoch: '+ str(epoch) + ' val_mae: ' + str(val_mae) + ' val_rmse: ' + str(val_rmse)+ '\n'
            f_log.write(str_log)
            f_log.close()

            my_test(test_loader, metadata, kf_filepath, config)

    plt.plot(loss_list)
    plt.ylabel('Loss')
    plt.xlabel("time")
    plt.savefig(kf_filepath+'/loss.png')
    plt.show()

def my_val(model, val_loader, device, scheduler):
    p_affinity = []
    y_affinity = []

    model.eval()
    loss_epoch = 0
    n = 0
    for data in val_loader:
        with torch.no_grad():
            data = set_gpu(data, device)
            predict = model(data)
            loss = F.mse_loss(predict, data[0].y)
            loss_epoch += loss.item()
            n += 1

            p_affinity.extend(predict.cpu().tolist())
            y_affinity.extend(data[0].y.cpu().tolist())


    scheduler.step(loss_epoch / n)

    affinity_err = metrics_reg(targets=y_affinity,predicts=p_affinity)

    return affinity_err


def my_test(test_loader, metadata, kf_filepath, config):
    p_affinity = []
    y_affinity = []

    m_state_dict = torch.load(kf_filepath+'best_model.pt')
    best_model = LBAPredictor(metadata=metadata, config=config, device=device).to(device)
    best_model.load_state_dict(m_state_dict)
    best_model.eval()

    for i, data in enumerate(test_loader, 0):
        with torch.no_grad():
            data = set_gpu(data, device)
            predict = best_model(data)
            p_affinity.extend(predict.cpu().tolist())
            y_affinity.extend(data[0].y.cpu().tolist())

    affinity_err = metrics_reg(targets=y_affinity,predicts=p_affinity)

    f_log = open(file=(kf_filepath+"/metrics_log.txt"), mode="a")
    str_log = 'mae: '+ str(affinity_err[0]) + ' rmse: '+ str(affinity_err[1]) + ' r: '+ str(affinity_err[2]) +' sd: '+ str(affinity_err[3]) + '\n'
    f_log.write(str_log)
    f_log.close()



if __name__ == '__main__':
    """ Please use the process.py file to preprocess the raw data and set up the training, validation, and test sets """

    with open("./configs/config.yml", 'r') as f:
        config = EasyDict(yaml.safe_load(f))

    print("loading data")
    train_set = PLBA_Dataset('file','./data/train.pkl')
    val_set = PLBA_Dataset('file', './data/valid.pkl')
    test_set = PLBA_Dataset('file','./data/test.pkl')
    metadata = train_set[0][1].metadata()

    model = LBAPredictor(metadata=metadata, config=config, device=device).to(device)

    batch_size = 1
    batch_vars = ["atom_coords", "seq"]
    train_loader = DataLoader(dataset=train_set, batch_size=batch_size, follow_batch=batch_vars)
    val_loader = DataLoader(dataset=val_set, batch_size=batch_size, follow_batch=batch_vars)
    test_loader = DataLoader(dataset=test_set, batch_size=batch_size, follow_batch=batch_vars)
    print("Preprocessing dataset")
    train_set = iterate_surface_precompute(train_loader, model.pmn.protein_surface_encoder, device)
    val_set = iterate_surface_precompute(val_loader, model.pmn.protein_surface_encoder, device)
    test_set = iterate_surface_precompute(test_loader, model.pmn.protein_surface_encoder, device)

    batch_size = 16
    batch_vars = ["atom_coords", "seq"]
    train_loader = DataLoader(dataset=train_set, batch_size=batch_size, follow_batch=batch_vars, shuffle=True)
    val_loader = DataLoader(dataset=val_set, batch_size=batch_size, follow_batch=batch_vars, shuffle=True)
    test_loader = DataLoader(dataset=test_set, batch_size=batch_size, follow_batch=batch_vars, shuffle=False)
    
    filepath = './output/'
    my_train(train_loader, val_loader, test_loader, filepath, model, config)







