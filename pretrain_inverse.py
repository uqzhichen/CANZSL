import torch
import torch.nn.functional as F
from torch.autograd import Variable
import torch.autograd as autograd
import torch.optim as optim
import torch.nn.init as init


from termcolor import cprint
from time import gmtime, strftime
import numpy as np
import argparse
import os
import glob
import random
import json

from dataset_GBU import FeatDataLayer, DATA_LOADER
from models import _netD2_att,_netG2_att, _param

parser = argparse.ArgumentParser()
parser.add_argument('--dataset', default='CUB1', help='FLO')
parser.add_argument('--dataroot', default='data/GBU/',
                    help='path to dataset')
parser.add_argument('--matdataset', default=True, help='Data in matlab format')
parser.add_argument('--image_embedding', default='res101')
parser.add_argument('--class_embedding', default='att')
parser.add_argument('--preprocessing', action='store_true', default=False,
                    help='enbale MinMaxScaler on visual features')
parser.add_argument('--standardization', action='store_true', default=False)
parser.add_argument('--validation', action='store_true', default=False, help='enable cross validation mode')

parser.add_argument('--gpu', default='1', type=str, help='index of GPU to use')
parser.add_argument('--exp_idx', default='', type=str, help='exp idx')
parser.add_argument('--manualSeed', type=int, help='manual seed')
parser.add_argument('--resume', default='',  type=str, help='the model to resume') #./out/Best_model_ZSL_Acc_51.64.tar

parser.add_argument('--z_dim',  type=int, default=100, help='dimension of the random vector z')
parser.add_argument('--disp_interval', type=int, default=20)
parser.add_argument('--save_interval', type=int, default=200)
parser.add_argument('--evl_interval',  type=int, default=40)

opt = parser.parse_args()
print('Running parameters:')
print(json.dumps(vars(opt), indent=4, separators=(',', ':')))

os.environ['CUDA_VISIBLE_DEVICES'] = opt.gpu

""" hyper-parameter  """
opt.GP_LAMBDA = 10    # Gradient penalty lambda
opt.CENT_LAMBDA  = 5
opt.REG_W_LAMBDA = 0.001
opt.Adv_LAMBDA = 1

opt.lr = 0.0001
opt.batchsize = 1024  # 512

""" hyper-parameter for testing"""
opt.nSample = 60  # number of fake feature for each class
opt.Knn = 20      # knn: the value of K


if opt.manualSeed is None:
    opt.manualSeed = random.randint(1, 10000)
print("Random Seed: ", opt.manualSeed)
random.seed(opt.manualSeed)
torch.manual_seed(opt.manualSeed)
torch.cuda.manual_seed_all(opt.manualSeed)


def train():
    param = _param()
    dataset = DATA_LOADER(opt)
    param.X_dim = dataset.feature_dim

    data_layer = FeatDataLayer(dataset.train_label.numpy(), dataset.train_feature.numpy(), opt)

    netG2 = _netG2_att(opt, dataset.text_dim, dataset.feature_dim).cuda()
    netG2.apply(weights_init)
    print(netG2)
    netD2 = _netD2_att(dataset.text_dim, dataset.train_cls_num).cuda()
    netD2.apply(weights_init)
    print(netD2)


    exp_info = 'GBU_{}_PretrainG2D2'.format(opt.dataset)
    exp_params = 'Eu{}_Rls{}'.format(opt.CENT_LAMBDA, opt.REG_W_LAMBDA)

    out_dir  = 'out/{:s}'.format(exp_info)
    out_subdir = 'out/{:s}/{:s}'.format(exp_info, exp_params)
    if not os.path.exists('out'):
        os.mkdir('out')
    if not os.path.exists(out_dir):
        os.mkdir(out_dir)
    if not os.path.exists(out_subdir):
        os.mkdir(out_subdir)

    cprint(" The output dictionary is {}".format(out_subdir), 'red')
    log_dir = out_subdir + '/log_{:s}_{}.txt'.format(exp_info, opt.exp_idx)
    with open(log_dir, 'w') as f:
        f.write('Training Start:')
        f.write(strftime("%a, %d %b %Y %H:%M:%S +0000", gmtime()) + '\n')

    start_step = 0

    if opt.resume:
        if os.path.isfile(opt.resume):
            print("=> loading checkpoint '{}'".format(opt.resume))
            checkpoint = torch.load(opt.resume)
            netG2.load_state_dict(checkpoint['state_dict_G2'])
            netD2.load_state_dict(checkpoint['state_dict_D2'])

            start_step = checkpoint['it']
            print(checkpoint['log'])
        else:
            print("=> no checkpoint found at '{}'".format(opt.resume))
    nets = [netD2 , netD2]

    optimizerD2 = optim.Adam(netD2.parameters(), lr=opt.lr, betas=(0.5, 0.9))
    optimizerG2 = optim.Adam(netG2.parameters(), lr=opt.lr, betas=(0.5, 0.9))

    for it in range(start_step, 3000+1):
        """D2"""
        for _ in range(5):
            blobs = data_layer.forward()
            feat_data = blobs['data']             # image data
            labels = blobs['labels'].astype(int)  # class labels
            text_feat = np.array([ dataset.train_att[i,:] for i in labels])
            text_feat = Variable(torch.from_numpy(text_feat.astype('float32'))).cuda()
            X = Variable(torch.from_numpy(feat_data)).cuda()
            y_true = Variable(torch.from_numpy(labels.astype('int'))).cuda()
            z = Variable(torch.randn(opt.batchsize, opt.z_dim)).cuda()
            z2 = Variable(torch.randn(opt.batchsize, param.z_dim)).cuda()

            # real loss
            D2_real, C2_real = netD2(text_feat)
            D2_loss_real = torch.mean(D2_real)
            C2_loss_real = F.cross_entropy(C2_real, y_true)
            DC2_loss = -D2_loss_real +  C2_loss_real
            DC2_loss.backward()

            # fake loss
            text_sample = netG2(z,X).detach()
            D2_fake,C2_fake = netD2(text_sample)
            D2_loss_fake = torch.mean(D2_fake)
            C2_loss_fake = F.cross_entropy(C2_fake, y_true)
            DC2_loss = D2_loss_fake   + C2_loss_fake
            DC2_loss.backward()

            # train with gradient penalty (WGAN_GP)
            grad_penalty = calc_gradient_penalty(netD2, text_feat.data, text_sample.data)
            grad_penalty.backward()
            Wasserstein_D = D2_loss_real - D2_loss_fake
            optimizerD2.step()
            reset_grad(nets)

        """G2"""
        for _ in range(1):
            blobs = data_layer.forward()
            feat_data = blobs['data']  # image data
            labels = blobs['labels'].astype(int)  # class labels
            text_feat = np.array([dataset.train_att[i, :] for i in labels])
            text_feat = Variable(torch.from_numpy(text_feat.astype('float32'))).cuda()
            X = Variable(torch.from_numpy(feat_data)).cuda()
            y_true = Variable(torch.from_numpy(labels.astype('int'))).cuda()
            z = Variable(torch.randn(opt.batchsize, opt.z_dim)).cuda()


            text_sample = netG2(z, X)
            D2_fake, C2_fake = netD2(text_sample)
            G2_loss = torch.mean(D2_fake)
            C2_loss_fake = F.cross_entropy(C2_fake, y_true)
            GC2_loss = -G2_loss + C2_loss_fake

            # ||W||_2 regularization
            reg_loss = Variable(torch.Tensor([0.0])).cuda()
            if opt.REG_W_LAMBDA != 0:
                for name, p in netG2.named_parameters():
                    if 'weight' in name:
                        reg_loss += p.pow(2).sum()
                reg_loss.mul_(opt.REG_W_LAMBDA)

            all_loss = GC2_loss + 0.1*reg_loss
            all_loss.backward()
            optimizerG2.step()
            reset_grad(nets)

        if it % opt.disp_interval == 0 and it:
            acc_real = (np.argmax(C2_real.data.cpu().numpy(), axis=1) == y_true.data.cpu().numpy()).sum() / float(y_true.data.size()[0])
            acc_fake = (np.argmax(C2_fake.data.cpu().numpy(), axis=1) == y_true.data.cpu().numpy()).sum() / float(y_true.data.size()[0])

            log_text = 'Iter-{}; Was_D: {:.3f};  reg_ls: {:.3f}; G_loss: {:.3f}; D_loss_real: {:.3f};' \
                       ' D_loss_fake: {:.3f}; rl: {:.2f}%; fk: {:.2f}%; c_rl: {:.2f}; c_fk: {:.2f}'\
                        .format(it, Wasserstein_D.item(),  reg_loss.item(),
                                G2_loss.item(), D2_loss_real.item(), D2_loss_fake.item(),
                                acc_real * 100, acc_fake * 100, C2_loss_real.item(), C2_loss_fake.item())

            print(log_text)
            with open(log_dir, 'a') as f:
                f.write(log_text+'\n')


        if it % opt.save_interval == 0 and it:
            save_model(it,netG2,netD2, opt.manualSeed, log_text,
                       out_subdir + '/Iter_{:d}.tar'.format(it))
            cprint('Save model to ' + out_subdir + '/Iter_{:d}.tar'.format(it), 'red')


def save_model(it,netG2,netD2, random_seed, log, fout):
    torch.save({
        'it': it + 1,
        'state_dict_G2': netG2.state_dict(),
        'state_dict_D2': netD2.state_dict(),
        'random_seed': random_seed,
        'log': log,
    }, fout)


def weights_init(m):
    classname = m.__class__.__name__
    if 'Linear' in classname:
        init.xavier_normal(m.weight.data)
        init.constant(m.bias, 0.0)


def reset_grad(nets):
    for net in nets:
        net.zero_grad()


def label2mat(labels, y_dim):
    c = np.zeros([labels.shape[0], y_dim])
    for idx, d in enumerate(labels):
        c[idx, d] = 1
    return c


def calc_gradient_penalty(netD, real_data, fake_data):
    alpha = torch.rand(opt.batchsize, 1)
    alpha = alpha.expand(real_data.size())
    alpha = alpha.cuda()

    interpolates = alpha * real_data + ((1 - alpha) * fake_data)
    interpolates = interpolates.cuda()
    interpolates = autograd.Variable(interpolates, requires_grad=True)

    disc_interpolates, _ = netD(interpolates)

    gradients = autograd.grad(outputs=disc_interpolates, inputs=interpolates,
                              grad_outputs=torch.ones(disc_interpolates.size()).cuda(),
                              create_graph=True, retain_graph=True, only_inputs=True)[0]
    gradient_penalty = ((gradients.norm(2, dim=1) - 1) ** 2).mean() * opt.GP_LAMBDA
    return gradient_penalty


if __name__ == "__main__":
    train()

