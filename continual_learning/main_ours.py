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
from torchvision.utils import save_image, make_grid
import warnings

warnings.filterwarnings("ignore")

sys.path.append('../')
from utils.misc import get_network, get_dataset, get_eval_pool, get_loops, get_time, get_daparam, mkdir, TensorDataset, inf_train_gen, load_yaml, write_yaml, to_numpy, concat
from utils.ops import evaluate_synset, match_loss, epoch
from utils.augmentation import DiffAugment, ParamDiffAug
from opacus.utils.uniform_sampler import UniformWithReplacementSampler
from opacus.privacy_engine import PrivacyEngine
from rdp_accountant import compute_sigma, compute_epsilon


def parse_arguments():
    parser = argparse.ArgumentParser(description='Parameter Processing')
    ## general experiment config
    parser.add_argument('--exp_name', '-name', type=str, default='ours_default', help='set up path for storing the results')
    parser.add_argument('--dataset', type=str, default='MNIST', help='dataset')
    parser.add_argument('--data_root', type=str, default='../data', help='path for the data')
    parser.add_argument('--random_seed', '-s', type=int, default=1000, help='random seed')
    parser.add_argument('--only_eval', action='store_true', help='If only perform evaluation')
    parser.add_argument('--load_checkpoint', action='store_true', help='If continue training from checkpoints')

    ## cl training config
    parser.add_argument('--model', type=str, default='ConvNet', help='model')
    parser.add_argument('--num_eval', type=int, default=3, help='the number of evaluating randomly initialized models')
    parser.add_argument('--nb_cl', type=int, default=2, help='num of classes per task')

    ## hyperparameters for dataset distillation
    parser.add_argument('--method', type=str, default='DC', help='DC/DSA')
    parser.add_argument('--spc', type=int, default=10, help='image(s) per class')
    parser.add_argument('--epoch_eval_train', type=int, default=300, help='epochs to train a model with synthetic data')
    parser.add_argument('--Iteration', type=int, default=200, help='training iterations')
    parser.add_argument('--batch_loop', type=int, default=10, help='img loops (for updating synthetic data)')
    parser.add_argument('--outer_loop', type=int, default=-1, help='outer loops (for updating synthetic data)')
    parser.add_argument('--inner_loop', type=int, default=-1, help='inner loops (for updating network with synthetic data)')
    parser.add_argument('--lr_img', type=float, default=0.1, help='learning rate for updating synthetic images')
    parser.add_argument('--lr_net', type=float, default=0.01, help='learning rate for updating network parameters')
    parser.add_argument('--batch_real', type=int, default=256, help='batch size for real data')
    parser.add_argument('--batch_train', type=int, default=256, help='batch size for training networks')
    parser.add_argument('--dsa_strategy', type=str, default='None', help='differentiable Siamese augmentation strategy')
    parser.add_argument('--dis_metric', type=str, default='gm', help='distance metric')

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
    if args.load_checkpoint or args.only_eval:
        log_file = open(os.path.join(save_path, 'log.txt'), "a")
    else:
        log_file = open(os.path.join(save_path, 'log.txt'), "w")
    sys.stdout = log_file  # save output to logfile
    use_cuda = torch.cuda.is_available()
    args.device = 'cuda' if use_cuda else 'cpu'
    args.dsa_param = ParamDiffAug()
    args.dsa = True if args.method == 'DSA' else False
    outer_loop, inner_loop = get_loops(args.spc)  # obtain default setting (will be overwritten if specified)
    if args.outer_loop == -1:
        args.outer_loop = outer_loop
    if args.inner_loop == -1:
        args.inner_loop = inner_loop

    ### Random seed
    args.random_seed = random.randint(1, 10 ^ 5) if args.random_seed is None else args.random_seed
    random.seed(args.random_seed)
    torch.manual_seed(args.random_seed)
    np.random.seed(args.random_seed)
    if use_cuda:
        torch.cuda.manual_seed_all(args.random_seed)

    ### Load data
    channel, im_size, num_classes, class_names, mean, std, dst_train, dst_test, testloader = get_dataset(args.dataset, data_path)
    X_train_total = dst_train.data 
    X_valid_total = dst_test.data 
    Y_train_total = to_numpy(dst_train.targets)
    Y_valid_total = to_numpy(dst_test.targets)

    order_list = load_yaml(os.path.join('configs', args.dataset + '_default.yml'))['order']
    order = np.array(order_list)
    X_valid_cumuls = []
    X_train_cumuls = []
    Y_valid_cumuls = []
    Y_train_cumuls = []
    n_experiences = int(num_classes / args.nb_cl)

    ### Placeholders
    trainset = dst_train
    testset = dst_test
    evalset = copy.deepcopy(dst_test)

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
    data_save = []
    print('Hyper-parameters: \n', args.__dict__)
    print('Evaluation model pool: ', model_eval_pool)

    ### Initialize the synthetic data from random noise/load from checkpoint
    iter_start = 0
    task_start = 0
    image_syn = torch.randn(size=(num_classes * args.spc, channel, im_size[0], im_size[1]), dtype=torch.float, requires_grad=True, device=args.device)
    label_syn = torch.tensor([np.ones(args.spc) * i for i in range(num_classes)], dtype=torch.long, requires_grad=False, device=args.device).view(-1)  # [0,0,0, 1,1,1, ..., 9,9,9]
    label_syn_total = np.concatenate([np.ones(args.spc) * i for i in range(num_classes)])
    if args.load_checkpoint or args.only_eval:
        checkpoint_path = os.path.join(save_path, 'checkpoint.pt')
        print('load synthetic data from checkpoint', checkpoint_path)
        checkpoint = torch.load(checkpoint_path)
        iter_start = checkpoint['iter'] + 1
        task_start = checkpoint['task_id']
        image_syn.data = checkpoint['image_syn'].to(args.device)
        label_syn.data = checkpoint['label_syn'].to(args.device)
        print('iterstart: {}'.format(iter_start))
    else:
        print('initialize synthetic data from random noise')

    ### Training
    criterion = nn.CrossEntropyLoss(reduction='mean').to(args.device)
    print('%s training begins' % get_time())
    for task_id in range(task_start, n_experiences):
        ## Get dataloader
        actual_cl = order[task_id * args.nb_cl: (task_id + 1) * args.nb_cl]
        indices_train_10 = np.array([i in order[task_id * args.nb_cl: (task_id + 1) * args.nb_cl] for i in to_numpy(Y_train_total)])
        indices_test_10 = np.array([i in order[task_id * args.nb_cl: (task_id + 1) * args.nb_cl] for i in to_numpy(Y_valid_total)])
        X_train = X_train_total[indices_train_10]
        X_valid = X_valid_total[indices_test_10]
        X_valid_cumuls.append(X_valid)
        X_train_cumuls.append(X_train)
        X_valid_cumul = concat(X_valid_cumuls)
        X_train_cumul = concat(X_train_cumuls)
        Y_train = Y_train_total[indices_train_10]
        Y_valid = Y_valid_total[indices_test_10]
        Y_valid_cumuls.append(Y_valid)
        Y_train_cumuls.append(Y_train)
        Y_valid_cumul = concat(Y_valid_cumuls)
        Y_train_cumul = concat(Y_train_cumuls)
        map_Y_train = torch.tensor([order_list.index(i) for i in Y_train])
        map_Y_valid_cumul = torch.tensor([order_list.index(i) for i in Y_valid_cumul])

        trainset.data = X_train
        trainset.targets = Y_train
        sampling_rate = args.batch_real / len(X_train)
        uniform_sampler = UniformWithReplacementSampler(num_samples=len(X_train), sample_rate=sampling_rate)
        trainloader = torch.utils.data.DataLoader(trainset, batch_sampler=uniform_sampler, pin_memory=True)
        inf_loader = inf_train_gen(trainloader)

        testset.data = X_valid
        testset.targets = Y_valid
        testloader = torch.utils.data.DataLoader(testset, batch_size=256, shuffle=False, num_workers=0)

        evalset.data = X_valid_cumul
        evalset.targets = map_Y_valid_cumul
        evalloader = torch.utils.data.DataLoader(evalset, batch_size=256, shuffle=False, num_workers=0)
        evalclasses = order[:(task_id + 1) * args.nb_cl]
        indices_syneval_10 = np.array([i in evalclasses for i in label_syn_total])
        indices_syntrain_10 = np.array([i in actual_cl for i in label_syn_total])
        map_label_syn = torch.tensor([order_list.index(i) for i in label_syn_total[indices_syneval_10]])

        ## Initial new optimizer for each task
        optimizer_img = torch.optim.SGD([image_syn, ], lr=args.lr_img, momentum=0.5)  # optimizer_img for synthetic data

        ### Compute sigma given #iterations and target privacy level (epsilon is compute w.r.t. sub-datasets for each task)
        if args.enable_privacy:
            iters = args.Iteration * args.outer_loop * args.batch_loop
            if args.sigma > 0:
                noise_multiplier = args.sigma
                print('Debugging, use fixed noise multiplier:', noise_multiplier)
            else:
                noise_multiplier = compute_sigma(args.target_epsilon, sampling_rate, iters, args.target_delta)
                print(f'Task{task_id}, Targer epsilon:{args.target_epsilon}, Noise multiplier{noise_multiplier}')

        for it in range(iter_start, args.Iteration + 1):
            ## Evaluate (Continual Learning setting)
            if it in eval_it_pool:
                for model_eval in model_eval_pool:
                    print('-------------------------\nEvaluation\nmodel_train = %s, model_eval = %s, iteration = %d' % (args.model, model_eval, it))
                    if args.dsa:
                        args.epoch_eval_train = 1000
                        args.dc_aug_param = None
                        print('DSA augmentation strategy: \n', args.dsa_strategy)
                        print('DSA augmentation parameters: \n', args.dsa_param.__dict__)
                    else:
                        args.dc_aug_param = get_daparam(args.dataset, args.model, model_eval, args.spc)  # This augmentation parameter set is only for DC method. It will be muted when args.dsa is True.
                        print('DC augmentation parameters: \n', args.dc_aug_param)

                    if args.dsa or args.dc_aug_param['strategy'] != 'none':
                        args.epoch_eval_train = 1000  # Training with data augmentation needs more epochs.
                    else:
                        args.epoch_eval_train = 300

                    accs = []
                    for it_eval in range(args.num_eval):
                        net_eval = get_network(model_eval, channel, len(evalclasses), im_size).to(args.device)  # get a random model
                        image_syn_eval = copy.deepcopy(image_syn.detach())
                        image_syn_eval = image_syn_eval[indices_syneval_10]
                        label_syn_eval = map_label_syn
                        net_eval, acc_train, acc_test = evaluate_synset(it_eval, net_eval, image_syn_eval, label_syn_eval, evalloader, args)
                        accs.append(acc_test)
                    print('Evaluate %d random %s, mean = %.4f std = %.4f\n-------------------------' % (len(accs), model_eval, np.mean(accs), np.std(accs)))

                    if it == args.Iteration:  # record the final results
                        accs_all_exps[model_eval][task_id] += accs

                ''' visualize and save '''
                save_name = os.path.join(save_path, 'task%d_%s_%dspc_iter%d.png' % (task_id, args.model, args.spc, it))
                image_syn_vis = copy.deepcopy(image_syn.detach().cpu())
                for ch in range(channel):
                    image_syn_vis[:, ch] = image_syn_vis[:, ch] * std[ch] + mean[ch]
                image_syn_vis[image_syn_vis < 0] = 0.0
                image_syn_vis[image_syn_vis > 1] = 1.0
                save_image(image_syn_vis, save_name, nrow=args.spc)  # Trying normalize = True/False may get better visual effects.

            ### Train synthetic data
            net = get_network(args.model, channel, num_classes, im_size).to(args.device)  # get a random model
            net_shadow = copy.deepcopy(net)  # Used for obtain DP real gradient (shadow is necessary as otherwise the hooks will cause problems)
            net.train()
            net_shadow.train()
            criterion = criterion
            net_parameters = list(net.parameters())
            net_shadow_parameters = list(net_shadow.parameters())
            optimizer_net_grad = torch.optim.SGD(net_shadow.parameters(), lr=args.lr_net)  # optimizer_net for obtaining dp grad
            optimizer_net = torch.optim.SGD(net.parameters(), lr=args.lr_net)  # optimizer_net for update model
            loss_avg = 0
            args.dc_aug_param = None  # Mute the DC augmentation when training synthetic data.

            ### Initialize for DP
            if args.enable_privacy:
                ### Initialize privacy engine
                privacy_engine = PrivacyEngine(net_shadow, sample_rate=sampling_rate, alphas=[1 + x / 10.0 for x in range(1, 100)] + list(range(12, 64)), noise_multiplier=noise_multiplier, max_grad_norm=args.max_norm)
                privacy_engine.attach(optimizer_net_grad)

            for ol in range(args.outer_loop):  # (~targeting at global trajectory matching)
                ### Optimize synthetic data
                for _ in range(args.batch_loop):  # sample multiple batches of real data and obtain gradients given the same model parameter (~target at local behavior matching)
                    img_real, lab_real = next(inf_loader)
                    img_real = img_real.to(args.device)
                    lab_real = lab_real.to(args.device)
                    img_syn = image_syn[indices_syntrain_10]
                    lab_syn = label_syn[indices_syntrain_10]

                    if args.dsa:
                        seed = int(time.time() * 1000) % 100000
                        img_real = DiffAugment(img_real, args.dsa_strategy, seed=seed, param=args.dsa_param)
                        img_syn = DiffAugment(img_syn, args.dsa_strategy, seed=seed, param=args.dsa_param)

                    ## Compute real_gradient
                    if args.enable_privacy:
                        net_shadow.load_state_dict(net.state_dict())  # synchronize the current model parameter (net -> net_shadow)
                        net_shadow.zero_grad()
                        output_real = net_shadow(img_real)
                        loss_real = criterion(output_real, lab_real)
                        loss_real.backward()
                        optimizer_net_grad.step()  # this step compute the DP noisy gradient on net_shadow
                        gw_real = list((p.grad.detach().clone() for p in net_shadow_parameters))
                    else:
                        net.zero_grad()
                        output_real = net(img_real)
                        loss_real = criterion(output_real, lab_real)
                        gw_real = torch.autograd.grad(loss_real, net_parameters)
                        gw_real = list((_.detach().clone() for _ in gw_real))

                    ## Compute fake_gradient and matching loss
                    net.zero_grad()
                    output_syn = net(img_syn)
                    loss_syn = criterion(output_syn, lab_syn)
                    gw_syn = torch.autograd.grad(loss_syn, net_parameters, create_graph=True)
                    loss = match_loss(gw_syn, gw_real, args)

                    ## Update image
                    optimizer_img.zero_grad()
                    loss.backward()
                    optimizer_img.step()
                    loss_avg += loss.item()

                if ol == args.outer_loop - 1:
                    break

                ### Update network (#inner_loop epochs on the current synthetic set)
                image_syn_train, label_syn_train = copy.deepcopy(image_syn.detach()), copy.deepcopy(label_syn.detach())  # avoid any unaware modification
                dst_syn_train = TensorDataset(image_syn_train, label_syn_train)
                trainloader = torch.utils.data.DataLoader(dst_syn_train, batch_size=args.batch_train, shuffle=True, num_workers=0)
                for il in range(args.inner_loop):
                    epoch('train', trainloader, net, optimizer_net, criterion, args, aug=True if args.dsa else False)

            if it == args.Iteration:  ## record the final results
                data_save.append([copy.deepcopy(image_syn.detach().cpu()), copy.deepcopy(label_syn.detach().cpu())])
                torch.save({'data': data_save, 'accs_all_exps': accs_all_exps}, os.path.join(save_path, 'res_%s_%s_%s_%dspc.pt' % (args.method, args.dataset, args.model, args.spc)))

            if it % 50 == 0:
                torch.save({'image_syn': copy.deepcopy(image_syn.detach().cpu()), 'label_syn': copy.deepcopy(label_syn.detach().cpu()), 'task_id': task_id, 'iter': it}, os.path.join(save_path, 'checkpoint.pt'))

    print('\n==================== Final Results ====================\n')
    for key in model_eval_pool:
        for task_id in range(n_experiences):
            accs = accs_all_exps[key][task_id]
            acc_avg = np.mean(accs) * 100
            acc_cum[key][task_id] = float(acc_avg)
            print('Task %d, Run %d eval experiments, train on %s, evaluate %d random %s, mean  = %.2f%%  std = %.2f%%' % (task_id, args.num_eval, args.model, len(accs), key, np.mean(accs) * 100, np.std(accs) * 100))
    write_yaml(acc_cum, os.path.join(save_path, 'accs.yml'))
    log_file.close()


if __name__ == '__main__':
    main()
