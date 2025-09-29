import os
import argparse
import torch
from torch_geometric.loader import DataLoader
from utils import build_optimizer, standardize_cdft
from model import SchNetAvg, SchNetNuc, ZSchNet, ZSchNet_CDFT
from train_funcs import train, test
from config import args
from dataset import Dataset
import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import r2_score
import copy
from scipy.stats import sem


def ten_fold(args, save_path):
    print('args:', args.__dict__)
    #
    ten_fold_record = {
        'fold': [],
        'r_2': [],
        'rmse': [],
        'ratio_2': [],
        'ratio_1': []
    }
    ten_fold_pred_record = {
        'Name': [],
        'N': [],
        'pred': [],
        'fold': []
    }
    # split all_data by nucleophile type
    dataset_path = 'resources/raw_dataset.pt'
    dataset = Dataset(data_path=dataset_path, cutoff=args.cutoff)  # list containing graphs
    all_data = dataset.get_data()
    dataset1 = []  # nucleophile types excluding C-I and N-SP
    dataset1_nuc_type = []
    dataset2 = []  # nucleophile types C-I and N-SP
    dataset2_nuc_type = []
    for data in all_data:
        nuc_type = data.nuc_type
        if nuc_type not in ['C-I', 'N-SP']:
            dataset1.append(data)
            dataset1_nuc_type.append(nuc_type)
        else:
            dataset2.append(data)
            dataset2_nuc_type.append(nuc_type)

    skf1 = StratifiedKFold(n_splits=10, shuffle=True, random_state=42)  # to split dataset1
    skf2 = StratifiedKFold(n_splits=2, shuffle=True, random_state=42)  # to split dataset2
    skf2_split = list(skf2.split(np.zeros(len(dataset2)), dataset2_nuc_type))
    for fold, (train_idx, val_idx) in enumerate(skf1.split(np.zeros(len(dataset1)), dataset1_nuc_type)):
        fold_save_path = os.path.join(save_path, f'fold{fold}')  # results/SchNet/0/fold0/
        if not os.path.exists(fold_save_path):
            os.mkdir(fold_save_path)
        train_set = copy.deepcopy([dataset1[i] for i in train_idx])
        val_set = copy.deepcopy([dataset1[j] for j in val_idx])
        # add C-I and N-SP to the first two folds
        if fold < 2:
            train_idx2 = skf2_split[fold][0]
            val_idx2 = skf2_split[fold][1]
            train_set += copy.deepcopy([dataset2[i] for i in train_idx2])
            val_set += copy.deepcopy([dataset2[j] for j in val_idx2])

        # save original train_set and test_set; must deepcopy otherwise pyg raise 'GlobalStorage' error
        torch.save(copy.deepcopy(train_set), os.path.join(fold_save_path, 'train_set.pt'))
        torch.save(copy.deepcopy(val_set), os.path.join(fold_save_path, 'val_set.pt'))

        #
        standardize_cdft(train_set, val_set)
        train_loader = DataLoader(train_set, batch_size=args.batch_size, shuffle=True)
        val_loader = DataLoader(val_set, batch_size=args.batch_size)
        print('len train_loader.dataset:', len(train_loader.dataset))
        print('len val_loader.dataset:', len(val_loader.dataset))

        # build model
        if args.model_name == 'SchNetAvg':
            model = SchNetAvg(
                n_filters=args.n_filters,
                n_interactions=args.n_interactions,
                u_max=args.cutoff,
                output_dim=args.output_dim
            )
        elif args.model_name == 'SchNetNuc':
            model = SchNetNuc(
                n_filters=args.n_filters,
                n_interactions=args.n_interactions,
                u_max=args.cutoff,
                output_dim=args.output_dim
            )
        elif args.model_name == 'ZSchNet':
            model = ZSchNet(
                n_filters=args.n_filters,
                n_interactions=args.n_interactions,
                u_max=args.cutoff,
                output_dim=args.output_dim
            )
        elif args.model_name == 'ZSchNet_CDFT':
            model = ZSchNet_CDFT(
                n_filters=args.n_filters,
                n_interactions=args.n_interactions,
                u_max=args.cutoff,
                output_dim=args.output_dim
            )
        else:
            raise Exception('Unknown model name')
        model.to(args.device)
        print(model)

        # construct optimizer
        scheduler, optimizer = build_optimizer(args, model.parameters())

        # train
        history, best_model = train(model, [train_loader, val_loader],
                                    optimizer, scheduler, args)

        # save model
        weight_name = f"{args.model_name}_cutoff{args.cutoff}_T{args.n_interactions}_fold{fold}" + '_weight.pt'
        model_path = os.path.join(fold_save_path, weight_name)
        torch.save(best_model.state_dict(), model_path)

        # eval best_model
        mse, mae, rmse, mol_ls, y, pred = test(best_model, val_loader, args)
        r_2 = r2_score(y.cpu().numpy(), pred.cpu().numpy())
        ratio_2 = (torch.abs(y - pred) <= 2.).sum() / y.size(0)
        ratio_1 = (torch.abs(y - pred) <= 1.).sum() / y.size(0)
        #
        ten_fold_record['fold'].append(fold)
        ten_fold_record['r_2'].append(r_2)
        ten_fold_record['rmse'].append(rmse.item())
        ten_fold_record['ratio_2'].append(ratio_2.item())
        ten_fold_record['ratio_1'].append(ratio_1.item())
        ten_fold_pred_record['Name'].extend(mol_ls)
        ten_fold_pred_record['N'].extend(y.cpu().tolist())
        ten_fold_pred_record['pred'].extend(pred.cpu().tolist())
        ten_fold_pred_record['fold'].extend([fold] * len(mol_ls))

        print('cutoff: {}'.format(args.cutoff))
        print('MSE: {:.8f}'.format(mse))
        print('MAE: {:.8f}'.format(mae))
        print('RMSE: {:.8f}'.format(rmse))
        print('R_2: {:.5f}'.format(r_2))
        print('Ratio_2: {:.5f}'.format(ratio_2))
        print('Ratio_1: {:.5f}'.format(ratio_1))
        print('#' * 40)
        print('\n\n')

    df = pd.DataFrame(ten_fold_record)
    df.to_csv(os.path.join(save_path, '10-fold_result.csv'), index=False)  # results/SchNet/0/10-fold_result.csv
    df2 = pd.DataFrame(ten_fold_pred_record)
    df2.to_csv(os.path.join(save_path, '10-fold_pred.csv'), index=False)  # results/SchNet/0/10-fold_pred.csv

    return np.mean(ten_fold_record['r_2']), np.mean(ten_fold_record['rmse'])


if __name__ == '__main__':
    parser = argparse.ArgumentParser("Get 10-fold cross-validation result.")
    parser.add_argument('--model', type=str, default='ZSchNet_CDFT',
                        help="specify the model type from SchNetAvg/SchNetNuc/ZSchNet/ZSchNet_CDFT "
                             "(default: ZSchNet_CDFT)")
    parser.add_argument("--cutoff", type=int, default=5,
                        help="distance cutoff to define atom neighbors (default: 5); "
                             "choose one from [3, 5, 10, 20]")
    parser.add_argument('--cuda', dest='is_cuda', action='store_true',
                        help="use a GPU to train the model")
    parser.add_argument('--runs', type=int, default=10,
                        help="runs number of the 10-fold CV (default: 10)")
    parser.set_defaults(is_cuda=False)

    input_args = parser.parse_args()

    #
    args.model_name = input_args.model
    args.cutoff = input_args.cutoff
    if input_args.is_cuda:
        args.device = 'cuda'
    else:
        args.device = 'cpu'
    args.cv_runs = input_args.runs

    # --------------10 runs of 10-fold CV--------------
    SAVE_DIR = f'./results'
    # models
    result_record = {}
    r2_ls = []
    rmse_ls = []
    for times in range(args.cv_runs):
        save_dir = os.path.join(SAVE_DIR, args.model_name, f'{times}')  # results/SchNet/0/
        if not os.path.exists(save_dir):
            os.makedirs(save_dir)
        #
        r2, rmse = ten_fold(args, save_dir)
        r2_ls.append(r2)
        rmse_ls.append(rmse)
    #
    result_record[args.model_name] = {
        'r2': np.mean(r2_ls),
        'r2_sem': sem(r2_ls),
        'rmse': np.mean(rmse_ls),
        'rmse_sem': sem(rmse_ls)
    }

    result_df = pd.DataFrame(result_record)
    result_df.to_csv(os.path.join(SAVE_DIR, f'{args.cv_runs}-runs_10-fold_results.csv'))

