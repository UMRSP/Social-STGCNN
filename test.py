import os
import math
import sys
import torch
import numpy as np
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
import pickle
import argparse
import glob
import torch.distributions.multivariate_normal as torchdist
from utils import * 
from metrics import * 
from model import social_stgcnn
import copy
from time import time
# Determine if CUDA-MPS MAC or CPU
device = torch.device("cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu")

def test(KSTEPS=20):
    global loader_test, model
    model.eval()
    ade_bigls = []
    fde_bigls = []
    raw_data_dict = {}
    step = 0 
    
    for batch in loader_test: 
        start = time()
        step += 1
        # Get data and move tensors to the correct hardware device
        batch = [tensor.to(device) for tensor in batch]
        obs_traj, pred_traj_gt, obs_traj_rel, pred_traj_gt_rel, non_linear_ped,\
         loss_mask, V_obs, A_obs, V_tr, A_tr = batch

        num_of_objs = obs_traj_rel.shape[1]

        # Forward
        V_obs_tmp = V_obs.permute(0, 3, 1, 2)
        V_pred, _ = model(V_obs_tmp, A_obs.squeeze())
        V_pred = V_pred.permute(0, 2, 3, 1)

        V_tr = V_tr.squeeze()
        A_tr = A_tr.squeeze()
        V_pred = V_pred.squeeze()
        num_of_objs = obs_traj_rel.shape[1]
        V_pred, V_tr = V_pred[:, :num_of_objs, :], V_tr[:, :num_of_objs, :]

        # Extract bi-variate parameters 
        sx = torch.exp(V_pred[:, :, 2]) # sx
        sy = torch.exp(V_pred[:, :, 3]) # sy
        corr = torch.tanh(V_pred[:, :, 4]) # corr
        
        # Initialize covariance matrix on the correct hardware device
        cov = torch.zeros(V_pred.shape[0], V_pred.shape[1], 2, 2).to(device)
        cov[:, :, 0, 0] = sx * sx
        cov[:, :, 0, 1] = corr * sx * sy
        cov[:, :, 1, 0] = corr * sx * sy
        cov[:, :, 1, 1] = sy * sy
        mean = V_pred[:, :, 0:2]
        
        mvnormal = torchdist.MultivariateNormal(mean, cov)

        ### Rel to abs 
        ade_ls = {}
        fde_ls = {}
        V_x = seq_to_nodes(obs_traj.data.cpu().numpy().copy())
        V_x_rel_to_abs = nodes_rel_to_nodes_abs(V_obs.data.cpu().numpy().squeeze().copy(),
                                                 V_x[0, :, :].copy())

        V_y = seq_to_nodes(pred_traj_gt.data.cpu().numpy().copy())
        V_y_rel_to_abs = nodes_rel_to_nodes_abs(V_tr.data.cpu().numpy().squeeze().copy(),
                                                 V_x[-1, :, :].copy())
        
        raw_data_dict[step] = {}
        raw_data_dict[step]['obs'] = copy.deepcopy(V_x_rel_to_abs)
        raw_data_dict[step]['trgt'] = copy.deepcopy(V_y_rel_to_abs)
        raw_data_dict[step]['pred'] = []

        for n in range(num_of_objs):
            ade_ls[n] = []
            fde_ls[n] = []

        for k in range(KSTEPS):
            V_pred = mvnormal.sample()
            V_pred_rel_to_abs = nodes_rel_to_nodes_abs(V_pred.data.cpu().numpy().squeeze().copy(),
                                                       V_x[-1, :, :].copy())
            raw_data_dict[step]['pred'].append(copy.deepcopy(V_pred_rel_to_abs))
            
            for n in range(num_of_objs):
                pred = [] 
                target = []
                obsrvs = [] 
                number_of = []
                pred.append(V_pred_rel_to_abs[:, n:n+1, :])
                target.append(V_y_rel_to_abs[:, n:n+1, :])
                obsrvs.append(V_x_rel_to_abs[:, n:n+1, :])
                number_of.append(1)

                ade_ls[n].append(ade(pred, target, number_of))
                fde_ls[n].append(fde(pred, target, number_of))
        
        for n in range(num_of_objs):
            ade_bigls.append(min(ade_ls[n]))
            fde_bigls.append(min(fde_ls[n]))
        print(time()-start)

    ade_ = sum(ade_bigls) / len(ade_bigls) if len(ade_bigls) > 0 else 0
    fde_ = sum(fde_bigls) / len(fde_bigls) if len(fde_bigls) > 0 else 0
    return ade_, fde_, raw_data_dict

# Ensure that multiprocessing on Windows works correctly
if __name__ == '__main__':
    print(f"Using device: {device}")
    
    paths = ['./checkpoint/*social-stgcnn*']
    KSTEPS = 20

    print("*" * 50)
    print('Number of samples:', KSTEPS)
    print("*" * 50)

    # Global tracking for overall averages
    overall_ade_ls = []
    overall_fde_ls = []

    for feta in range(len(paths)):
        path = paths[feta]
        exps = glob.glob(path)
        print('Models being tested are:', exps)

        for exp_path in exps:
            print("*" * 50)
            print("Evaluating model:", exp_path)

            model_path = exp_path + '/val_best.pth'
            args_path = exp_path + '/args.pkl'
            
            # Use 'rb' safely; skip if missing
            if not os.path.exists(args_path):
                print(f"Skipping {exp_path}, missing args.pkl")
                continue
                
            with open(args_path, 'rb') as f: 
                args = pickle.load(f)

            stats = exp_path + '/constant_metrics.pkl'
            if os.path.exists(stats):
                with open(stats, 'rb') as f: 
                    cm = pickle.load(f)
                print("Stats:", cm)

            # Data prep 
            obs_seq_len = args.obs_seq_len
            pred_seq_len = args.pred_seq_len
            data_set = './datasets/' + args.dataset + '/'

            dset_test = TrajectoryDataset(
                    data_set + 'test/',
                    obs_len=obs_seq_len,
                    pred_len=pred_seq_len,
                    skip=1, norm_lap_matr=True)

            loader_test = DataLoader(
                    dset_test,
                    batch_size=1, # Irrelative to args batch size parameter
                    shuffle=False,
                    num_workers=1)

            # Defining the model and mapping it dynamically to available hardware
            model = social_stgcnn(
                n_stgcnn=args.n_stgcnn, n_txpcnn=args.n_txpcnn,
                output_feat=args.output_size, seq_len=args.obs_seq_len,
                kernel_size=args.kernel_size, pred_seq_len=args.pred_seq_len
            ).to(device)
            
            # Load weights to the assigned device
            model.load_state_dict(torch.load(model_path, map_location=device))

            ade_ = 999999
            fde_ = 999999
            print("Testing ....")
            
            # Execute test
            ad, fd, raw_data_dic_ = test(KSTEPS=KSTEPS)
            
            ade_ = min(ade_, ad)
            fde_ = min(fde_, fd)
            
            overall_ade_ls.append(ade_)
            overall_fde_ls.append(fde_)
            print(f"ADE: {ade_:.4f}  FDE: {fde_:.4f}")

    print("*" * 50)
    if len(overall_ade_ls) > 0:
        print(f"Avg ADE: {sum(overall_ade_ls) / len(overall_ade_ls):.4f}")
        print(f"Avg FDE: {sum(overall_fde_ls) / len(overall_fde_ls):.4f}")
    else:
        print("No models were evaluated.")