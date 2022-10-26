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
import warnings
warnings.filterwarnings("ignore")
from torch.utils.data import DataLoader, TensorDataset
from opacus.utils.uniform_sampler import UniformWithReplacementSampler
from opacus.privacy_engine import PrivacyEngine

from cl_utils import Logger, train, test
sys.path.append('../')
from utils.misc import get_network, get_dataset, mkdir, load_yaml, write_yaml
from rdp_accountant import compute_sigma, compute_epsilon


def parse_arguments():
    parser = argparse.ArgumentParser(description='Parameter Processing')
    ## general experiment config
    parser.add_argument('--exp_name', '-name', type=str, default='dpsgd_default', help='path for storing the results')
    parser.add_argument('--dataset', type=str, default='MNIST', help='dataset')
    parser.add_argument('--data_root', type=str, default='../data', help='path for the data')
    parser.add_argument('--random_seed', '-s', type=int, default=1000, help='random seed')

    ## cl training config
    parser.add_argument('--model', type=str, default='ConvNet', help='model')
    parser.add_argument('--train_epoch', type=int, default=5, help='epochs for each task')
    parser.add_argument('--lr_net', type=float, default=0.01, help='learning rate for updating network parameters')
    parser.add_argument('--batch_size', type=int, default=256, help='batch size for real data')
    parser.add_argument('--nb_cl', type=int, default=2, help='num of classes per task')

    ## parameter specific for DP
    parser.add_argument('--enable_privacy', default=False, action='store_true', help='Enable private data generation')
    parser.add_argument('--target_epsilon', type=float, default=10, help='Epsilon DP parameter')
    parser.add_argument('--target_delta', type=float, default=1e-5, help='Delta DP parameter')
    parser.add_argument('--max_norm', type=float, default=0.1, help='The coefficient to clip the gradients')
    parser.add_argument('--sigma', type=float, default=0, help='Gaussian noise variance multiplier.')
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
    log_file = open(os.path.join(save_path, 'out.txt'), "w")
    sys.stdout = log_file
    title = args.dataset + '_' + args.model
    logger = Logger(os.path.join(save_path, 'log.txt'), title=title)
    logger.set_names(['Task ID', 'Epoch', 'Train Acc', 'Val Acc', 'Train Loss', 'Val Loss'])

    ### Random seed
    args.random_seed = random.randint(1, 10 ^ 5) if args.random_seed is None else args.random_seed
    random.seed(args.random_seed)
    torch.manual_seed(args.random_seed)
    np.random.seed(args.random_seed)
    if use_cuda:
        torch.cuda.manual_seed_all(args.random_seed)

    ### Obtain Data stream
    channel, im_size, num_classes, class_names, mean, std, dst_train, dst_test, testloader = get_dataset(args.dataset, data_path)
    X_train_total = dst_train.data
    Y_train_total = dst_train.targets
    X_valid_total = dst_test.data
    Y_valid_total = dst_test.targets
    order_list = load_yaml(os.path.join('configs', args.dataset + '_default.yml'))['order']
    order = np.array(order_list)
    X_valid_cumuls = []
    X_train_cumuls = []
    Y_valid_cumuls = []
    Y_train_cumuls = []
    n_experiences = int(num_classes / args.nb_cl)

    # placeholder for datasets
    trainset = dst_train
    testset = dst_test
    evalset = copy.deepcopy(dst_test)

    ### Init
    model = get_network(args.model, channel, num_classes, im_size).to(args.device)
    criterion = nn.CrossEntropyLoss()
    acc_cum = {}
    nsamples_cum = {}

    ### Training
    for task_id in range(n_experiences):
        ## get dataloader
        actual_cl = order[task_id * args.nb_cl: (task_id + 1) * args.nb_cl]
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

        trainset.data = X_train
        trainset.targets = Y_train
        sampling_rate = args.batch_size / len(X_train)
        uniform_sampler = UniformWithReplacementSampler(num_samples=len(X_train), sample_rate=sampling_rate)
        trainloader = torch.utils.data.DataLoader(trainset, batch_sampler=uniform_sampler, pin_memory=True)

        testset.data = X_valid
        testset.targets = Y_valid
        testloader = torch.utils.data.DataLoader(testset, batch_size=256, shuffle=False, num_workers=2)

        evalset.data = X_valid_cumul
        evalset.targets = Y_valid_cumul
        evalloader = torch.utils.data.DataLoader(evalset, batch_size=256, shuffle=False, num_workers=2)
        evalclasses = order[:(task_id + 1) * args.nb_cl]

        ## Set optimizer on the temporary model (avoid error of the DP optimizer)
        model_tmp = get_network(args.model, channel, num_classes, im_size).to(args.device)  # get a random model
        model_tmp.load_state_dict(model.state_dict())
        optimizer = torch.optim.SGD(model_tmp.parameters(), lr=args.lr_net)

        ### Private training (epsilon is compute w.r.t. sub-datasets for each task)
        if args.enable_privacy:
            iters = args.train_epoch * len(trainloader)
            sampling_prob = sampling_rate
            if args.sigma > 0:
                noise_multiplier = args.sigma
                print('Debugging, use pre-defined sigma=', noise_multiplier)
            else:
                noise_multiplier = compute_sigma(args.target_epsilon, sampling_prob, iters, args.target_delta)
                print(f'Target epsilon{args.target_epsilon}, delta{args.target_delta}: sigma:{noise_multiplier}')
            privacy_engine = PrivacyEngine(model_tmp, sample_rate=sampling_prob, alphas=[1 + x / 10.0 for x in range(1, 100)] + list(range(12, 64)), noise_multiplier=noise_multiplier, max_grad_norm=args.max_norm, )
            privacy_engine.attach(optimizer)

        ### Do DP-SGD training
        for ep in range(args.train_epoch):
            train_loss, train_acc = train(trainloader, model_tmp, criterion, optimizer, args.device)
            test_loss, test_acc = test(testloader, model_tmp, criterion, args.device)
            logger.append([task_id, ep, train_acc, test_acc, train_loss, test_loss])

        ### Copy temporary model to the final model and eval
        model.load_state_dict(model_tmp.state_dict())
        print('=' * 30)
        print('Eval current task ')
        print('Task %d, Train acc: %f, Test acc: %f' % (task_id, train_acc, test_acc))

        ## Average Acc
        correct = 0
        total_samples = 0
        for (test_x, test_y) in evalloader:
            model.eval()
            test_x = test_x.to(args.device)
            test_y = test_y.to(args.device)
            test_logits = model(test_x)
            correct += test_y.eq(test_logits.argmax(dim=1)).sum().item()
            total_samples += len(test_x)
        acc = correct * 100 / total_samples
        acc_cum['task%d' % task_id] = acc
        nsamples_cum['task%d' % task_id] = total_samples

    print('\n==================== Final Results ====================\n')
    for task_id in range(n_experiences):
        print('Task id up to %d, Test acc: %f, Samples: %f' % (task_id, acc_cum['task%d' % task_id], nsamples_cum['task%d' % task_id]))
    write_yaml(acc_cum, os.path.join(save_path, 'accs.yml'))


if __name__ == '__main__':
    main()
