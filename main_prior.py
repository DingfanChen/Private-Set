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
from utils.misc import get_loops, get_dataset, get_network, get_eval_pool, get_daparam, get_time, TensorDataset, mkdir, inf_train_gen
from utils.ops import evaluate_synset, match_loss, epoch
from utils.augmentation import DiffAugment, ParamDiffAug
from utils.gan_models import GeneratorDCGAN
from rdp_accountant import compute_sigma, compute_epsilon
from opacus.utils.uniform_sampler import UniformWithReplacementSampler
from opacus.privacy_engine import PrivacyEngine


def parse_arguments():
    parser = argparse.ArgumentParser(description='Parameter Processing')
    ## general experiment config
    parser.add_argument('--exp_name', '-name', type=str, default='prior_default', help='set up path for storing the results')
    parser.add_argument('--dataset', type=str, default='MNIST', help='dataset')
    parser.add_argument('--data_root', type=str, default='data', help='path for the data')
    parser.add_argument('--random_seed', '-s', type=int, default=1000, help='random seed')
    parser.add_argument('--only_eval', action='store_true', help='If only perform evaluation')
    parser.add_argument('--load_checkpoint', action='store_true', help='If continue training from checkpoints')

    ## hyperparameters for dataset distillation
    parser.add_argument('--method', type=str, default='DC', help='DC/DSA')
    parser.add_argument('--model', type=str, default='ConvNet', help='model') # Note that BN is not compatible with DP
    parser.add_argument('--spc', type=int, default=1, help='image(s) per class')
    parser.add_argument('--eval_mode', type=str, default='S', help='eval_mode')  # S: the same to training model, M: multi architectures,  W: net width, D: net depth, A: activation function, P: pooling layer, N: normalization layer,
    parser.add_argument('--num_exp', type=int, default=1, help='the number of experiments')
    parser.add_argument('--num_eval', type=int, default=1, help='the number of evaluating randomly initialized models')
    parser.add_argument('--epoch_eval_train', type=int, default=300, help='epochs to train a model with synthetic data')
    parser.add_argument('--Iteration', type=int, default=1000, help='training iterations')
    parser.add_argument('--batch_loop', type=int, default=10, help='batch loops (for updating synthetic data)')
    parser.add_argument('--outer_loop', type=int, default=2, help='outer loops (for updating synthetic data)')
    parser.add_argument('--inner_loop', type=int, default=10, help='inner loops (for updating network with synthetic data)')
    parser.add_argument('--lr_img', type=float, default=0.01, help='learning rate for updating synthetic images')
    parser.add_argument('--lr_net', type=float, default=0.01, help='learning rate for updating network parameters')
    parser.add_argument('--batch_real', type=int, default=256, help='batch size for real data')
    parser.add_argument('--batch_train', type=int, default=256, help='batch size for training networks')
    parser.add_argument('--dsa_strategy', type=str, default='None', help='differentiable Siamese augmentation strategy')
    parser.add_argument('--data_path', type=str, default='data', help='dataset path')
    parser.add_argument('--dis_metric', type=str, default='gm', help='distance metric')

    ## hyperparameters for GAN setting
    parser.add_argument('--gan_model', type=str, default='DCGAN', help='gan model')
    parser.add_argument('--z_dim', type=int, default=10, help='latent dim')
    parser.add_argument('--latent_type', type=str, default='normal', help='latent type')

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
    sys.stdout = log_file
    use_cuda = torch.cuda.is_available()
    args.device = 'cuda' if use_cuda else 'cpu'
    args.dsa_param = ParamDiffAug()
    args.dsa = True if args.method == 'DSA' else False

    ### Random seed
    args.random_seed = random.randint(1, 10 ^ 5) if args.random_seed is None else args.random_seed
    random.seed(args.random_seed)
    torch.manual_seed(args.random_seed)
    np.random.seed(args.random_seed)
    if use_cuda:
        torch.cuda.manual_seed_all(args.random_seed)

    ### Eval setting
    eval_it_pool = np.concatenate((np.arange(10, args.Iteration + 1, 100), [args.Iteration])).tolist() if args.eval_mode == 'S' else [args.Iteration]  # The list of iterations when we evaluate models and record results.
    if args.Iteration not in eval_it_pool:
        eval_it_pool = np.append(eval_it_pool, args.Iteration)
    model_eval_pool = get_eval_pool(args.eval_mode, args.model, args.model)

    ### Data loader (use uniform sampler)
    channel, im_size, num_classes, class_names, mean, std, dst_train, dst_test, testloader = get_dataset(args.dataset, data_path)
    sample_rate = args.batch_real / len(dst_train)
    uniform_sampler = UniformWithReplacementSampler(num_samples=len(dst_train), sample_rate=sample_rate, )
    real_loader = torch.utils.data.DataLoader(dst_train, batch_sampler=uniform_sampler, pin_memory=True, )
    inf_loader = inf_train_gen(real_loader)

    ### Fix noise for optimization and visualization (fix_noise: used for optimization; nonopt_noise: used for visualization only, not being optimized for during training)
    nrows = 20
    if args.latent_type == 'normal':
        fix_noise = torch.randn(args.spc * num_classes, args.z_dim).to(args.device)
        nonopt_noise = torch.randn(nrows * num_classes, args.z_dim).to(args.device)
    elif args.latent_type == 'bernoulli':
        p = 0.5
        bernoulli = torch.distributions.Bernoulli(torch.tensor([p]))
        fix_noise = bernoulli.sample((args.spc * num_classes, args.z_dim)).view(nrows, args.z_dim).to(args.device)
        nonopt_noise = bernoulli.sample((nrows * num_classes, args.z_dim)).view(nrows, args.z_dim).to(args.device)
    else:
        raise NotImplementedError
    fix_label = torch.tensor([np.ones(args.spc) * i for i in range(num_classes)], dtype=torch.long, requires_grad=False, device=args.device).view(-1)
    nonopt_label = torch.tensor([np.ones(nrows) * i for i in range(num_classes)], dtype=torch.long, requires_grad=False, device=args.device).view(-1)

    ### Record performances of all experiments
    results_dict = dict()
    final_accs_dict = dict()
    for key in model_eval_pool:
        final_accs_dict[key] = []
        results_dict[key] = []
    results_dict['iter'] = np.sort(np.unique(eval_it_pool))
    epsilon_it = []
    data_save = []

    ### Compute sigma given #iterations and target privacy level
    if args.enable_privacy:
        k = args.Iteration * args.outer_loop * args.batch_loop
        epsilon = args.target_epsilon
        delta = args.target_delta
        if args.sigma > 0:
            noise_multiplier = args.sigma
            print('Debugging, use pre-defined sigma=', noise_multiplier)
        else:
            noise_multiplier = compute_sigma(epsilon, args.batch_real / len(dst_train), k, delta)
            print(f'eps,delta = ({epsilon},{delta}) ==> Noise level sigma=', noise_multiplier)

    for exp in range(args.num_exp):
        print('\n================== Exp %d ==================\n ' % exp)
        print('Hyper-parameters: \n', args.__dict__)
        print('Evaluation model pool: ', model_eval_pool)

        ### Organize the real dataset
        indices_class = [[] for c in range(num_classes)]
        images_all = [torch.unsqueeze(dst_train[i][0], dim=0) for i in range(len(dst_train))]
        labels_all = [dst_train[i][1] for i in range(len(dst_train))]
        for i, lab in enumerate(labels_all):
            indices_class[lab].append(i)
        images_all = torch.cat(images_all, dim=0).to(args.device)
        labels_all = torch.tensor(labels_all, dtype=torch.long, device=args.device)
        for ch in range(channel):
            print('real images channel %d, mean = %.4f, std = %.4f' % (ch, torch.mean(images_all[:, ch]), torch.std(images_all[:, ch])))

        ### Initialize generator, optimizer, criterion
        if args.gan_model == 'DCGAN':
            netG = GeneratorDCGAN(z_dim=args.z_dim, num_classes=num_classes, latent_type=args.latent_type, outact=nn.Identity())
        else:
            raise NotImplementedError
        netG = netG.to(args.device)
        optimizer_img = torch.optim.Adam(netG.parameters(), lr=args.lr_img)  # optimizer_img for generator (synthetic data)
        criterion = nn.CrossEntropyLoss(reduction='mean').to(args.device)
        iter_start = 0
        if args.load_checkpoint or args.only_eval:
            checkpoint = torch.load(os.path.join(save_path, 'checkpoint.pt'))
            netG.load_state_dict(checkpoint['netG'])
            optimizer_img.load_state_dict(checkpoint['optimizer_img'])
            image_syn_eval = checkpoint['image']
            label_syn_eval = checkpoint['label']
            iter_start = checkpoint['iter'] + 1
            print('iterstart: {}'.format(iter_start))
        else:
            print('initialize from scratch')

        ### Only perform evaluation
        if args.only_eval:
            for model_eval in model_eval_pool:
                print('-------------------------\nEvaluation\nmodel_train = %s, model_eval = %s, iteration = %d' % (
                    args.model, model_eval, iter_start))
                if args.dsa:
                    args.dc_aug_param = None
                    print('DSA augmentation strategy: \n', args.dsa_strategy)
                    print('DSA augmentation parameters: \n', args.dsa_param.__dict__)
                else:
                    args.dc_aug_param = get_daparam(args.dataset, args.model, model_eval, args.spc)  # This augmentation parameter set is only for DC method. It will be muted when args.dsa is True.
                    print('DC augmentation parameters: \n', args.dc_aug_param)

                accs = []
                for it_eval in range(args.num_eval):
                    net_eval = get_network(model_eval, channel, num_classes, im_size).to(args.device)  # get a random model
                    image_syn_eval.to(args.device)
                    label_syn_eval.to(args.device)
                    _, acc_train, acc_test = evaluate_synset(it_eval, net_eval, image_syn_eval, label_syn_eval, testloader, args)
                    accs.append(acc_test)
                print('Evaluate %d random %s, mean = %.4f std = %.4f\n-------------------------' % (
                    len(accs), model_eval, np.mean(accs), np.std(accs)))

                return

        ### Training
        print('%s training begins' % get_time())
        for it in range(iter_start, args.Iteration + 1):

            if it in eval_it_pool: ### evaluation
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
                        net_eval = get_network(model_eval, channel, num_classes, im_size).to(args.device)  # get a random model
                        netG.eval()
                        image_syn_eval = netG(fix_noise, fix_label)  ## use the GAN structure only as an image prior
                        # image_syn_eval, label_syn_eval = netG.sample(args.num_samples, args.device)  ## test the inherent generalization ability of GAN (by including completely unseen latent codes)
                        image_syn_eval = image_syn_eval.view(args.spc * num_classes, channel, im_size[0], im_size[1])
                        label_syn_eval = fix_label
                        _, acc_train, acc_test = evaluate_synset(it_eval, net_eval, image_syn_eval, label_syn_eval, testloader, args)
                        accs.append(acc_test)
                    print('Evaluate %d random %s, mean = %.4f std = %.4f\n-------------------------' % (len(accs), model_eval, np.mean(accs), np.std(accs)))
                    results_dict[model_eval].append(accs)
                    if it == args.Iteration:  # record the final results
                        final_accs_dict[model_eval] += accs

                ### Visualize and save
                image_syn_eval = image_syn_eval.cpu()
                image_syn_vis_nonopt = netG(nonopt_noise, nonopt_label).view(nrows * num_classes, channel, im_size[0], im_size[1]).cpu() # obtain random samples (unseen latent code)
                for ch in range(channel):
                    image_syn_eval[:, ch] = image_syn_eval[:, ch] * std[ch] + mean[ch]
                    image_syn_vis_nonopt[:, ch] = image_syn_vis_nonopt[:, ch] * std[ch] + mean[ch]
                image_syn_eval[image_syn_eval < 0] = 0.0
                image_syn_eval[image_syn_eval > 1] = 1.0
                image_syn_vis_nonopt[image_syn_vis_nonopt < 0] = 0.0
                image_syn_vis_nonopt[image_syn_vis_nonopt > 1] = 1.0
                save_name = os.path.join(save_path, 'eval_%s_%s_exp%d_iter%d.png' % (args.method, args.model, exp, it))
                save_image(image_syn_eval, save_name, nrow=args.spc)
                save_name = os.path.join(save_path, 'nonopt_%s_%s_exp%d_iter%d.png' % (args.method, args.model, exp, it))
                save_image(image_syn_vis_nonopt, save_name, nrow=nrows)

            ### Optimize synthetic data
            net = get_network(args.model, channel, num_classes, im_size).to(args.device)  # get a random model
            net_shadow = copy.deepcopy(net)  # Used for obtain DP real gradient (shadow is necessary as otherwise the hooks will cause problems)
            criterion = criterion
            net_parameters = list(net.parameters())
            net_shadow_parameters = list(net_shadow.parameters())
            optimizer_net_grad = torch.optim.SGD(net_shadow.parameters(), lr=args.lr_net)  # optimizer for obtaining DP real gradient
            optimizer_net = torch.optim.SGD(net.parameters(), lr=args.lr_net)  # optimizer_net for update model
            loss_avg = 0
            args.dc_aug_param = None  # Mute the DC augmentation when training synthetic data.

            ### Initialize for DP
            if args.enable_privacy:
                ### Initialize privacy engine
                privacy_engine = PrivacyEngine(net_shadow, sample_size=len(dst_train), batch_size=args.batch_real, alphas=[1 + x / 10.0 for x in range(1, 100)] + list(range(12, 64)), noise_multiplier=noise_multiplier, max_grad_norm=args.max_norm, )
                privacy_engine.attach(optimizer_net_grad)

            for ol in range(args.outer_loop): # (~targeting at global trajectory matching)
                ### Optimize synthetic data
                for _ in range(args.batch_loop):  # sample multiple batches of real data and obtain gradients given the same model parameter (~target at local behavior matching)
                    img_real, lab_real = next(inf_loader)
                    img_real = img_real.to(args.device)
                    lab_real = lab_real.to(args.device)

                    ## Generate fake data
                    netG.train()
                    img_syn = netG(fix_noise, fix_label)
                    lab_syn = fix_label
                    img_syn = img_syn.view(fix_noise.shape[0], channel, im_size[0], im_size[1])

                    if args.dsa:
                        seed = int(time.time() * 1000) % 100000
                        img_real = DiffAugment(img_real, args.dsa_strategy, seed=seed, param=args.dsa_param)
                        img_syn = DiffAugment(img_syn, args.dsa_strategy, seed=seed, param=args.dsa_param)

                    ## compute real_gradient
                    if args.enable_privacy:
                        net.train()
                        net_shadow.load_state_dict(net.state_dict())  # synchronize the current model parameter (net -> net_shadow)
                        net_shadow.train()
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
                netG.eval()
                image_syn_train = netG(fix_noise, fix_label)
                image_syn_train = image_syn_train.view(fix_noise.shape[0], channel, im_size[0], im_size[1])
                label_syn_train = fix_label
                dst_syn_train = TensorDataset(image_syn_train, label_syn_train)
                trainloader = torch.utils.data.DataLoader(dst_syn_train, batch_size=args.batch_train, shuffle=True, num_workers=0)
                for il in range(args.inner_loop):
                    epoch('train', trainloader, net, optimizer_net, criterion, args, aug=True if args.dsa else False)

            if it in eval_it_pool:
                if args.enable_privacy:
                    k = it * args.batch_loop * args.outer_loop
                    epsilon = compute_epsilon(noise_multiplier, args.batch_real / len(dst_train), k, args.target_delta)
                else:
                    epsilon = np.inf
                loss_test, acc_test = epoch('test', testloader, net, optimizer_net, criterion, args, aug=False)
                epsilon_it.append(epsilon)
                print('{} iter={}, loss={}, acc={}, ep={}'.format(get_time(), it, loss_avg, acc_test, epsilon))
                netG.eval()
                image_syn_eval = netG(fix_noise, fix_label).view(args.spc * num_classes, channel, im_size[0], im_size[1])
                label_syn_eval = fix_label
                torch.save({'fix_noise': fix_noise, 'fix_label': fix_label, 'image': image_syn_eval, 'label': label_syn_eval, 'netG': netG.state_dict(), 'optimizer_img': optimizer_img.state_dict(), 'iter': it}, os.path.join(save_path, 'checkpoint.pt'))

            if it % 100 == 0:
                results_dict['epsilon_it'] = np.array(epsilon_it)
                pickle.dump(results_dict, open(os.path.join(save_path, 'results.pkl'), 'wb'))
                netG.eval()
                image_syn_eval = netG(fix_noise, fix_label)
                label_syn_eval = fix_label
                image_syn_eval = image_syn_eval.view(args.spc * num_classes, channel, im_size[0], im_size[1])
                torch.save({'fix_noise': fix_noise, 'fix_label': fix_label, 'image': image_syn_eval, 'label': label_syn_eval, 'netG': netG.state_dict(), 'optimizer_img': optimizer_img.state_dict(), 'iter': it}, os.path.join(save_path, 'checkpoint.pt'))

            if it == args.Iteration:  ## record the final results
                data_save.append([copy.deepcopy(image_syn_eval.detach().cpu()), copy.deepcopy(label_syn_eval.detach().cpu())])
                torch.save({'data': data_save, 'final_accs_dict': final_accs_dict, }, os.path.join(save_path, 'res_%s_%s_%s_%dspc.pt' % (args.method, args.dataset, args.model, args.spc)))

    print('\n==================== Final Results ====================\n')
    for key in model_eval_pool:
        accs = final_accs_dict[key]
        print('Run %d experiments, train on %s, evaluate %d random %s, mean  = %.2f%%  std = %.2f%%' % (args.num_exp, args.model, len(accs), key, np.mean(accs) * 100, np.std(accs) * 100))
    results_dict['epsilon_it'] = np.array(epsilon_it)
    pickle.dump(results_dict, open(os.path.join(save_path, 'results.pkl'), 'wb'))
    log_file.close()


if __name__ == '__main__':
    main()
