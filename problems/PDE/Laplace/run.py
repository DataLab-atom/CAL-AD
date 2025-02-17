import os
import torch
import time
import numpy as np
import scipy.io as io
import math


from scipy.interpolate import Rbf
# from scipy.interpolate import interp2d
from torch.utils.data import Dataset, DataLoader

from mpl_toolkits.axes_grid1 import make_axes_locatable
from sklearn.decomposition import PCA
import argparse

# def ParseArgument():
#     parser = argparse.ArgumentParser(description='DeepONet')
#     parser.add_argument('--epochs', type=int, default=50000, metavar='N',
#                         help = 'number of epochs to train (default: 1000)')
#     parser.add_argument('--device', type=str, default='cuda', metavar='N',
#                         help = 'computing device (default: GPU)')
#     # parser.add_argument('--save-step', type=int, default=10000, metavar='N',
#     #                     help = 'save_step (default: 10000)')
#     parser.add_argument('--restart', type=int, default=0, metavar='N',
#                         help = 'if restart (default: 0)')
#     # parser.add_argument('--test_model', type=int, default=0, metavar='N',
#     #                     help = 'default training, testing as 1')

#     args = parser.parse_args()
#     return args

def to_numpy(input):
    if isinstance(input, torch.Tensor):
        return input.detach().cpu().numpy()
    elif isinstance(input, np.ndarray):
        return input
    else:
        raise TypeError('Unknown type of input, expected torch.Tensor or '\
            'np.ndarray, but got {}'.format(type(input)))

def static_main(model, epochs, device):
    ## hyperparameters
    #args = ParseArgument()
    # epochs = 50000
    # device = 'cuda'
    # save_step = args.save_step
    # test_model = args.test_model
    # if test_model == 1:
    #     load_path = "problems\DIMON\model.pt"

    PODMode = 10
    num_bc = 68 #204
    skip = 3
    dim_br1 = [PODMode*2, 100, 100, 100]
    dim_br2 = [num_bc, 150, 150, 150, 100] #150
    dim_tr = [2, 100, 100, 100]
    # lmd = 0 #1e-6

    num_train = 3300 ## assume 1000+500
    num_test = 200 ## note mesh coordinate in Laplace_test.mat save #900-909 
    # num_cases = num_train + num_test
    
    ## create folders
    # dump_train = './Predictions/Train/'
    # dump_test = './Predictions/Test/'
    # model_path = 'CheckPts/model_chkpts.pt'
    # os.makedirs(dump_train, exist_ok=True)
    # os.makedirs(dump_test, exist_ok=True)
    # os.makedirs('CheckPts', exist_ok=True)
    ## dataset main
#/root/AEL-P-SNE/outputs/dimon-PDE/2025-01-12_21-58-08
    datafile = "Laplace/dataset/Laplace_data.mat"
    dataset = io.loadmat(datafile)

    x = dataset["x_uni"]
    x_mesh = dataset["x_mesh_data"]
    dx = x_mesh - x ## shape: num_case, num_points, num_dim
    u = dataset["u_data"]
    u_bc = dataset["u_bc"]
    u_bc = u_bc[:, ::skip]

    ## dataset supp
    datafile_supp = "Laplace/dataset/Laplace_data_supp.mat"

    dataset_supp = io.loadmat(datafile_supp)

    x_mesh_supp = dataset_supp["x_mesh_data"]
    dx_supp = x_mesh_supp - x ## shape: num_case, num_points, num_dim
    u_supp = dataset_supp["u_data"]
    u_bc_supp = dataset_supp["u_bc"]
    u_bc_supp = u_bc_supp[:, ::skip]

    ## dataset supp 2
    datafile_supp2 = "Laplace/dataset/Laplace_data_supp2000.mat"
    dataset_supp2 = io.loadmat(datafile_supp2)

    x_mesh_supp2 = dataset_supp2["x_mesh_data"]
    dx_supp2 = x_mesh_supp2 - x ## shape: num_case, num_points, num_dim
    u_supp2 = dataset_supp2["u_data"]
    u_bc_supp2 = dataset_supp2["u_bc"]
    u_bc_supp2 = u_bc_supp2[:, ::skip]

    # mesh_coords_supp2 = dataset_supp2["mesh_coords"][0]
    # mesh_cells_supp2 = dataset_supp2["mesh_cells"][0]

    ## concat
    x_mesh = np.concatenate((x_mesh, x_mesh_supp, x_mesh_supp2), axis=0)
    dx = np.concatenate((dx, dx_supp, dx_supp2), axis=0)
    u = np.concatenate((u, u_supp, u_supp2), axis=0)
    u_bc = np.concatenate((u_bc, u_bc_supp, u_bc_supp2), axis=0)
    
    ## SVD
    dx_train = dx[:num_train]
    dx_test = dx[-num_test:]
    
    ## coefficient for training data
    dx1_train = dx_train[:, :, 0]
    dx2_train = dx_train[:, :, 1]

    pca_x = PCA(n_components=PODMode)
    pca_x.fit(dx1_train - dx1_train.mean(axis=0))
    coeff_x_train = pca_x.transform(dx1_train - dx1_train.mean(axis=0))
    coeff_x_test = pca_x.transform(dx_test[:, :, 0] - dx1_train.mean(axis=0))

    pca_y = PCA(n_components=PODMode)
    pca_y.fit(dx2_train - dx2_train.mean(axis=0))
    coeff_y_train = pca_y.transform(dx2_train - dx2_train.mean(axis=0))
    coeff_y_test = pca_y.transform(dx_test[:, :, 1] - dx2_train.mean(axis=0))
    

    f_train = np.concatenate((coeff_x_train, coeff_y_train), axis=1)
  
    ## coeffiient for testing data
    f_test = np.concatenate((coeff_x_test, coeff_y_test), axis=1)

    ##########################
    u_train = u[:num_train]
    u_test = u[-num_test:]
    f_bc_train = u_bc[:num_train, :]
    f_bc_test = u_bc[-num_test:, :]

    ## tensor
    u_test_tensor = torch.tensor(u_test, dtype=torch.float).to(device)
    f_test_tensor = torch.tensor(f_test, dtype=torch.float).to(device)
    f_bc_test_tensor = torch.tensor(f_bc_test, dtype=torch.float).to(device)

    u_train_tensor = torch.tensor(u_train, dtype=torch.float).to(device)
    f_train_tensor = torch.tensor(f_train, dtype=torch.float).to(device)
    f_bc_train_tensor = torch.tensor(f_bc_train, dtype=torch.float).to(device)

    x_tensor = torch.tensor(x, dtype=torch.float).to(device)

    ## initialization
    # model = model(dim_br1, dim_br2, dim_tr).to(device)
    model=model.to(device)
    model = model.float()
    optimizer = torch.optim.Adam(model.parameters(), lr = 0.001)
    
    train_loss = np.zeros((50000, 1))
    # test_loss = np.zeros((args.epochs, 1))
    # rel_l2_err_his = np.zeros((args.epochs, 1))

    ## training
    def train(epoch, f, f_bc, x, y):
        model.train()
        def closure():
            optimizer.zero_grad()

            ## L2 regularization
            # regularization_loss = 0
            # for param in model._branch.parameters():
            #     regularization_loss += torch.sum(param**2)
            # for param in model._trunk.parameters():
            #     regularization_loss += torch.sum(param**2)
            #for param in model._out_layer.parameters():
            #    regularization_loss += torch.sum(param**2)

            loss = model.loss(f, f_bc, x, y)
            train_loss[epoch, 0] = loss

            loss.backward()
            return loss
        optimizer.step(closure)

    ## Iterations
    # print('start training...', flush=True)
    # tic = time.time()
    for epoch in range(0, epochs):
        if epoch == 10000:
            optimizer = torch.optim.Adam(model.parameters(), lr = 0.0001)
        elif epoch == 90000:
            optimizer = torch.optim.Adam(model.parameters(), lr = 0.00001)
        elif epoch == epochs - 1000:
            optimizer = torch.optim.LBFGS(model.parameters())

        ## Training
        train(epoch, f_train_tensor, f_bc_train_tensor, x_tensor, u_train_tensor)
        ## Testing
        # loss_tmp = to_numpy(model.loss(f_test_tensor, f_bc_test_tensor, x_tensor, u_test_tensor))
        # test_loss[epoch, 0] = loss_tmp

        ## testing error
        # if epoch%100 == 0:
        #     print(f'Epoch: {epoch}, Train Loss: {train_loss[epoch, 0]:.6f}, Test Loss: {test_loss[epoch, 0]:.6f}', flush=True)
            
        # ## Save model
        # if (epoch+1)%save_step == 0: 
        #     torch.save({
        #         'model_state_dict': model.state_dict(),
        #         'optimizer_state_dict': optimizer.state_dict(),
        #     }, model_path)

    # toc = time.time()

    ##
    u_test_pred = to_numpy(model.forward(f_test_tensor, f_bc_test_tensor, x_tensor))
    # l2_err_test = ((u_test_pred - u_test)**2).mean(axis=1)
    u_train_pred = to_numpy(model.forward(f_train_tensor, f_bc_train_tensor, x_tensor))
    # l2_err_train = ((u_train_pred - u_train)**2).mean(axis=1)

    # rel_l2_err_train = (np.linalg.norm(u_train_pred - u_train, axis=1)/np.linalg.norm(u_train, axis=1))
    # rel_l2_err_test = (np.linalg.norm(u_test_pred - u_test, axis=1)/np.linalg.norm(u_test, axis=1))


    # print(f'total training time: {int((toc-tic)/60)} min', flush=True)
    '''
    np.savetxt('./train_loss.txt', train_loss)
    np.savetxt('./test_loss.txt', test_loss)
    np.savetxt('./rel_l2_test_his.txt', rel_l2_err_his)
    np.savetxt('./l2_err_test.txt', l2_err_test)
    np.savetxt('./l2_err_train.txt', l2_err_train)

    np.savetxt('./rel_l2_err_test.txt', rel_l2_err_test)
    np.savetxt('./rel_l2_err_train.txt', rel_l2_err_train)
    '''

    ## save train
    u_train_pred = model.forward(f_train_tensor, f_bc_train_tensor, x_tensor)
    u_pred = to_numpy(u_train_pred)
    u_true = u_train
        
    ##############
    ## save test
    u_test_pred = model.forward(f_test_tensor, f_bc_test_tensor, x_tensor)
    u_pred = to_numpy(u_test_pred)
    u_true = u_test
    # u_pred_save = []
    # abs_err_pred_save = []

    ## MSE err
    mean_abs_err = (abs(u_pred - u_true)).mean(axis=1)
    # print(mean_abs_err)

    rel_l2_err = np.linalg.norm(u_test - u_pred, axis=1)/np.linalg.norm(u_test, axis=1)
                        # np.mean()
    #np.savetxt("mean_abs_err.txt", mean_abs_err)
    #np.savetxt("rel_l2_err.txt", rel_l2_err)


    # if test_model == 1:
    #     # num2test ## 1500 in total and 

    #     save_lib = {}
    #     save_lib["u_pred_canonical_plot"] = u_pred_save
    #     save_lib["abs_err_pred_save"] = abs_err_pred_save
    #     save_lib["x_canonical"] = x
    #     save_lib["u_pred"] = u_pred
    #     save_lib["u_true"] = u_true

    #     save_lib["x_mesh"] = x_mesh_supp2[-num_test:]
    #     save_lib["mesh_coords"] = dataset_supp2["mesh_coords"][0][-num_test:]
    #     save_lib["mesh_cells"] = dataset_supp2["mesh_cells"][0][-num_test:]

    #     io.savemat("model_pred_vis.mat", save_lib)

    return mean_abs_err,rel_l2_err