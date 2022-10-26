import os
import sys
import time
import copy
import argparse
import numpy as np
import random
import pickle
import logging
import shutil
import torch
import torch.nn as nn
from torch.optim.lr_scheduler import StepLR
from torchvision import datasets, transforms
from torchvision.utils import save_image, make_grid
import warnings
warnings.filterwarnings("ignore")

from mean_embedding import weights_sphere, noisy_dataset_embedding, data_label_embedding, ConvCondGen
sys.path.append('../')
from utils.misc import get_network, get_dataset, get_eval_pool, get_time, get_daparam, mkdir, load_yaml, write_yaml
from utils.ops import evaluate_synset
from utils.augmentation import DiffAugment, ParamDiffAug
from rdp_accountant import compute_sigma, compute_epsilon


def parse_arguments():
    parser = argparse.ArgumentParser(description='Parameter Processing')
    ## general experiment config
    parser.add_argument('--exp_name', '-name', type=str, default='dpmerf_default', help='path for storing the results')
    parser.add_argument('--dataset', type=str, default='MNIST', help='dataset')
    parser.add_argument('--data_root', type=str, default='../data', help='path for the data')
    parser.add_argument('--random_seed', '-s', type=int, default=1000, help='random seed')
    parser.add_argument('--only_eval', action='store_true', help='If only perform evaluation')

    ## cl training config
    parser.add_argument('--model', type=str, default='ConvNet', help='model')
    parser.add_argument('--num_eval', type=int, default=3, help='the number of evaluating randomly initialized models')
    parser.add_argument('--nb_cl', type=int, default=2, help='num of classes per task')

    ## hyperparameters for method (use the original default values)
    parser.add_argument('--epoch_eval_train', type=int, default=20, help='epochs to train a model with synthetic data')
    parser.add_argument('--Iteration', type=int, default=3000, help='training iterations')
    parser.add_argument('--lr_img', type=float, default=0.1, help='learning rate for updating the generator network parameters')
    parser.add_argument('--lr_net', type=float, default=0.01, help='learning rate for updating classifier parameters')
    parser.add_argument('--batch_size', type=int, default=100, help='batch size for training')
    parser.add_argument('--batch_train', type=int, default=256, help='batch size for training the evaluation classifier')
    parser.add_argument('--dsa_strategy', type=str, default='None', help='differentiable Siamese augmentation strategy')
    parser.add_argument('--num_samples', type=int, default=6000, help='num of syn samples per class')
    parser.add_argument('--d_rff', type=int, default=1000, help='number of fourier features')
    parser.add_argument('--mmd_type', type=str, default='sphere', help='mmd type')
    parser.add_argument('--z_dim', type=int, default=5, help='latent dim')

    ## parameter specific for DP
    parser.add_argument('--enable_privacy', default=False, action='store_true', help='Enable private data generation')
    parser.add_argument('--target_epsilon', type=float, default=10, help='Epsilon DP parameter')
    parser.add_argument('--target_delta', type=float, default=1e-5, help='Delta DP parameter')
    parser.add_argument('--sigma', type=float, default=0, help='Gaussian noise variance multiplier.')
    parser.add_argument('--max_norm', type=float, default=0.1, help='The coefficient to clip the gradients')
    args = parser.parse_args()
    return args


def check_args(args):
    '''
    check and store the arguments as well as set up the save_dir
    :param args: arguments
    :return:
    '''
    ## set up save_dir
    save_dir = os.path.join(os.path.dirname(__file__), 'results', args.dataset, args.exp_name)
    mkdir(save_dir)

    ## store the parameters
    if not args.only_eval:
        with open(os.path.join(save_dir, 'params.txt'), 'w') as f:
            for k, v in vars(args).items():
                f.writelines(k + ":" + str(v) + "\n")
                print(k + ":" + str(v))
        pickle.dump(vars(args), open(os.path.join(save_dir, 'params.pkl'), 'wb'), protocol=2)

    ## store this script
    shutil.copy(os.path.realpath(__file__), save_dir)
    return args, save_dir


def main():
    ### General config
    args, save_path = check_args(parse_arguments())
    data_path = os.path.join(args.data_root, args.dataset)
    use_cuda = torch.cuda.is_available()
    args.device = 'cuda' if use_cuda else 'cpu'
    args.dsa = False
    log_file = open(os.path.join(save_path, 'log.txt'), "w")
    sys.stdout = log_file

    ### Random seed
    args.random_seed = random.randint(1, 10 ^ 5) if args.random_seed is None else args.random_seed
    random.seed(args.random_seed)
    torch.manual_seed(args.random_seed)
    np.random.seed(args.random_seed)
    if use_cuda:
        torch.cuda.manual_seed_all(args.random_seed)

    ### Load data
    channel, im_size, num_classes, class_names, mean, std, dst_train, dst_test, testloader = get_dataset(args.dataset, data_path)
    data_dim = channel * im_size[0] * im_size[1]
    if args.dataset == 'MNIST':
        dst_dpmerf = datasets.MNIST(data_path, train=True, download=True, transform=transforms.ToTensor())
        args.rff_sigma = 105
    elif args.dataset == 'FashionMNIST':
        dst_dpmerf = datasets.FashionMNIST(data_path, train=True, download=True, transform=transforms.ToTensor())
        args.rff_sigma = 127
    else:
        raise NotImplementedError
    X_train_total = dst_dpmerf.data
    Y_train_total = dst_dpmerf.targets
    X_valid_total = dst_test.data
    Y_valid_total = dst_test.targets
    order_list = load_yaml(os.path.join('configs', args.dataset + '_default.yml'))['order']
    order = np.array(order_list)
    X_valid_cumuls = []
    X_train_cumuls = []
    Y_valid_cumuls = []
    Y_train_cumuls = []
    n_experiences = int(num_classes / args.nb_cl)

    ### Placeholders
    trainset = dst_dpmerf
    testset = dst_test
    evalset = copy.deepcopy(dst_test)
    syn_data = []
    syn_target = []
    nrows = 20  # Set up for visualization
    fix_label = torch.tensor([np.ones(nrows) * i for i in range(args.nb_cl)], dtype=torch.long, requires_grad=False, device=args.device).view(-1)[:, None]

    ### Record performances of all experiments
    eval_it_pool = [args.Iteration]  # The list of iterations when we evaluate models and record results.
    model_eval_pool = get_eval_pool('S', args.model, args.model)
    accs_all_exps = dict()
    acc_cum = dict()
    for key in model_eval_pool:
        accs_all_exps[key] = dict()
        acc_cum[key] = dict()
        for task_id in range(n_experiences):
            accs_all_exps[key][task_id] = []
    print('Hyper-parameters: \n', args.__dict__)
    print('Evaluation model pool: ', model_eval_pool)

    ### Training
    criterion = nn.CrossEntropyLoss(reduction='mean').to(args.device)
    print('%s training begins' % get_time())
    for task_id in range(n_experiences):
        ## Get dataloader
        actual_cl = order[task_id * args.nb_cl: (task_id + 1) * args.nb_cl]
        actual_cl_list = actual_cl.tolist()
        indices_train_10 = np.array([i in order[task_id * args.nb_cl: (task_id + 1) * args.nb_cl] for i in Y_train_total.numpy()])
        indices_test_10 = np.array([i in order[task_id * args.nb_cl: (task_id + 1) * args.nb_cl] for i in Y_valid_total.numpy()])
        X_train = X_train_total[indices_train_10]
        X_valid = X_valid_total[indices_test_10]
        X_valid_cumuls.append(X_valid)
        X_train_cumuls.append(X_train)
        X_valid_cumul = torch.cat(X_valid_cumuls)
        X_train_cumul = torch.cat(X_train_cumuls)
        Y_train = Y_train_total[indices_train_10]
        Y_valid = Y_valid_total[indices_test_10]
        Y_valid_cumuls.append(Y_valid)
        Y_train_cumuls.append(Y_train)
        Y_valid_cumul = torch.cat(Y_valid_cumuls)
        Y_train_cumul = torch.cat(Y_train_cumuls)
        # map_Y_train = torch.tensor([order_list.index(i) for i in Y_train])
        map_Y_train = torch.tensor([actual_cl_list.index(i) for i in Y_train])
        map_Y_valid_cumul = torch.tensor([order_list.index(i) for i in Y_valid_cumul])

        ### Compute noisy mean embedding (epsilon is compute w.r.t. sub-datasets for each task)
        trainset.data = X_train
        trainset.targets = map_Y_train
        full_loader = torch.utils.data.DataLoader(trainset, batch_size=args.batch_size, shuffle=True, num_workers=0)
        if args.enable_privacy:
            k = 1 # one-shot static feature, i.e., iter=1
            noise_multiplier_emb = compute_sigma(args.target_epsilon, 1, k, args.target_delta)
        else:
            noise_multiplier_emb = 0
        print(f'Task{task_id}, Targer epsilon:{args.target_epsilon}, Noise multiplier{noise_multiplier_emb}')
        w_freq = weights_sphere(args.d_rff, data_dim, args.rff_sigma, args.device)
        noisy_emb = noisy_dataset_embedding(full_loader, w_freq, args.d_rff, args.device, args.nb_cl, noise_multiplier_emb, args.mmd_type, sum_frequency=25)

        ## Test set
        evalset.data = X_valid_cumul
        evalset.targets = map_Y_valid_cumul
        evalloader = torch.utils.data.DataLoader(evalset, batch_size=256, shuffle=False, num_workers=2)
        evalclasses = order[:(task_id + 1) * args.nb_cl]

        ## Init generator model training
        netG = ConvCondGen(d_code=args.z_dim, d_hid='200', n_labels=args.nb_cl, nc_str='16,8', ks_str='5,5')
        netG = netG.to(args.device)
        optimizer = torch.optim.Adam(netG.parameters(), lr=args.lr_img)
        scheduler = StepLR(optimizer, step_size=1, gamma=0.9)
        criterion = criterion
        args.dc_aug_param = None

        iter_start = 0
        for it in range(iter_start, args.Iteration + 1):
            if it in eval_it_pool: ## Evaluate (Continual Learning setting)
                netG.eval()
                label_syn_eval = torch.tensor([np.ones(args.num_samples) * i for i in range(args.nb_cl)], dtype=torch.long, requires_grad=False, device=args.device).view(-1)[:, None]
                code, _ = netG.get_code(len(label_syn_eval), labels=label_syn_eval, device=args.device)
                image_syn_eval = netG(code)
                image_syn_eval = image_syn_eval.view(len(label_syn_eval), channel, im_size[0], im_size[1])
                for ch in range(channel):
                    image_syn_eval[:, ch] = (image_syn_eval[:, ch] - mean[ch]) / std[ch]  # pre-processing (generator output of DP-Merf has range [0,1])
                label_syn_eval = actual_cl[label_syn_eval.cpu()]
                map_label_syn_eval = torch.tensor([order_list.index(i) for i in label_syn_eval])
                syn_data.append(image_syn_eval)
                syn_target.append(map_label_syn_eval)

                for model_eval in model_eval_pool:
                    print('-------------------------\nEvaluation\nmodel_train = %s, model_eval = %s, iteration = %d' % (args.model, model_eval, it))
                    if args.dsa:
                        args.epoch_eval_train = 60
                        args.dc_aug_param = None
                        args.dsa_param = ParamDiffAug()
                        print('DSA augmentation strategy: \n', args.dsa_strategy)
                        print('DSA augmentation parameters: \n', args.dsa_param.__dict__)
                    else:
                        args.dc_aug_param = get_daparam(args.dataset, args.model, model_eval, ipc=10)
                        print('DC augmentation parameters: \n', args.dc_aug_param)

                    if args.dsa or args.dc_aug_param['strategy'] != 'none':
                        args.epoch_eval_train = 60
                    else:
                        pass

                    accs = []
                    for it_eval in range(args.num_eval):
                        net_eval = get_network(model_eval, channel, len(evalclasses), im_size).to(args.device)
                        net_eval, acc_train, acc_test = evaluate_synset(it_eval, net_eval, torch.cat(syn_data), torch.cat(syn_target), evalloader, args)
                        accs.append(acc_test)
                    print('Evaluate %d random %s, mean = %.4f std = %.4f\n-------------------------' % (len(accs), model_eval, np.mean(accs), np.std(accs)))

                    if it == args.Iteration:  # record the final results
                        accs_all_exps[model_eval][task_id] += accs

                ''' visualize and save '''
                save_name = os.path.join(save_path, 'task%d_%s_iter%d.png' % (task_id, args.model, it))
                netG.eval()
                code, _ = netG.get_code(len(fix_label), labels=fix_label, device=args.device)
                image_syn_vis_fix = netG(code)
                image_syn_vis_fix = image_syn_vis_fix.view(nrows * args.nb_cl, channel, im_size[0], im_size[1])
                image_syn_vis_fix = image_syn_vis_fix.cpu()
                image_syn_vis_fix[image_syn_vis_fix < 0] = 0.0
                image_syn_vis_fix[image_syn_vis_fix > 1] = 1.0
                save_image(image_syn_vis_fix, save_name, nrow=nrows)

            ## learning rate scheduling
            if it % 599 == 0:
                scheduler.step()

            ## update generator
            netG.train()
            gen_code, gen_labels = netG.get_code(args.batch_size, args.device)
            gen_emb = data_label_embedding(netG(gen_code), gen_labels, w_freq, args.mmd_type)
            loss = torch.sum((noisy_emb - gen_emb) ** 2)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            if it % 50 == 0: # save checkpoints
                torch.save({'netG': netG.state_dict(), 'optimizer': optimizer.state_dict(), 'iter': it}, os.path.join(save_path, 'checkpoint.pt'))

    print('\n==================== Final Results ====================\n')
    for key in model_eval_pool:
        for task_id in range(n_experiences):
            accs = accs_all_exps[key][task_id]
            acc_avg = np.mean(accs) * 100
            acc_cum[key][task_id] = float(acc_avg)
            print('Task id up to %d, Run %d eval experiments, train on %s, evaluate %d random %s, mean  = %.2f%%  std = %.2f%%' % (task_id, args.num_eval, args.model, len(accs), key, np.mean(accs) * 100, np.std(accs) * 100))
    write_yaml(acc_cum, os.path.join(save_path, 'accs.yml'))
    log_file.close()

if __name__ == '__main__':
    main()
