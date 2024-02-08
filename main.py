import time

import argparse
import builtins
import os 
from datetime import timedelta
from tqdm import tqdm 

import torch
import torchvision 
from torch import autograd, nn, optim 
from torch.nn import functional as F

try:
    import wandb
except ImportError:
    wandb = None

import time

from models.discriminator import Discriminator
from models.generator import Generator
from utils import fid_score
from utils.CRDiffAug import CR_DiffAug
from utils.distributed import get_rank, reduce_loss_dict, synchronize

from dataset import get_dataset


def requires_grad(model, flag=True):
    for p in model.parameters():
        p.requires_grad = flag

def accumulate(model1, model2, decay=0.999):
    par1 = dict(model1.named_parameters())
    par2 = dict(model2.named_parameters())

    for k in par1.keys():
        par1[k].data.mul_(decay).add_(par2[k].data, alpha=1 - decay)

def sample_data(loader):
    while True:
        for batch in loader:
            yield batch

def d_logistic_loss(real_pred, fake_pred):
    assert type(real_pred) == type(fake_pred), "real_pred must be the same type as fake_pred"
    real_loss = F.softplus(-real_pred)
    fake_loss = F.softplus(fake_pred)
    return real_loss.mean() + fake_loss.mean()

def d_r1_loss(real_pred, real_img):
    grad_real, = autograd.grad(outputs=real_pred.sum(), inputs=real_img, create_graph=True)
    grad_penalty = grad_real.pow(2).reshape(grad_real.shape[0], -1).sum(1).mean()
    return grad_penalty

def g_nonsaturating_loss(fake_pred):
    loss = F.softplus(-fake_pred).mean()
    return loss

def get_time():
    return str(time.strftime("[%Y-%m-%d %H:%M:%S]", time.localtime()))

def log(d_loss, g_loss, r1_loss, G_lr, D_lr, step):
    wandb.log({'D_loss': d_loss, 'G_loss': g_loss, 'R1_loss': r1_loss, "G_lr": G_lr, "D_lr": D_lr}, step=step)

def save_checkpoint(g, d, g_ema, save_dir, step):
    checkpoint = {
        "g": g.state_dict(),
        "d": d.state_dict(),
        "g_ema": g_ema.state_dict(),
    }
    torch.save(checkpoint, os.path.join(save_dir, f"{str(step).zfill(6)}.pt"))

def train(trainloader, testloader, generator, discriminator, g_optim, d_optim, g_ema, args):
    wandb.init(sync_tensorboard=False, project="GanTraining", job_type="CleanRepo", config=args)

    save_dir = os.path.join(".", "logged_files", args.dataset, wandb.run.name)
    os.makedirs(save_dir, exist_ok=False)

    trainloader = sample_data(trainloader)

    accum = 0.5 ** (32 / (10 * 1000))
    l2_loss = torch.nn.MSELoss()

    r1_loss = torch.tensor(0.0, device=args.device)
    d_loss_val = 0
    g_loss_val = 0
    loss_dict = {}

    if args.distributed:
        g_module = generator.module
        d_module = discriminator.module
    else:
        g_module = generator
        d_module = discriminator

    print(" -- start training --")
    end = time.time()
    if args.ttur:
        args.G_lr = args.D_lr / 4
    if args.lr_decay: 
        lr_decay_per_step = args.G_lr / (args.iter - args.lr_decay_start_steps)

    for it in range(args.iter):
        wandb.log({"Progress" : it}, step=it)
        
        # train discriminator
        generator.train()
        if not args.lmdb:
            this_data = next(trainloader)
            real_img = this_data[0]
        else: 
            real_img = next(trainloader)
        real_img = real_img.to(args.device)

        requires_grad(generator, False)
        requires_grad(discriminator, True)

        noise = torch.randn((args.batch_real, args.start_dim)).cuda()

        fake_img, _ = generator(noise)
        fake_pred = discriminator(fake_img)
        real_pred = discriminator(real_img)
        d_loss = d_logistic_loss(real_pred, fake_pred) * args.gan_weight

        if args.bcr:
            real_img_cr_aug = CR_DiffAug(real_img)
            fake_img_cr_aug = CR_DiffAug(fake_img)
            real_pred_aug = discriminator(real_img_cr_aug)
            fake_pred_aug = discriminator(fake_img_cr_aug)
            d_loss += args.bcr_fake_lambda * l2_loss(fake_pred_aug, fake_pred) + args.bcr_real_lambda * l2_loss(real_pred_aug, real_pred)

        loss_dict["d"] = d_loss

        discriminator.zero_grad()
        d_loss.backward()
        nn.utils.clip_grad_norm_(discriminator.parameters(), 5.0)
        d_optim.step()

        d_regularize = it % args.d_reg_every == 0
        if d_regularize:
            real_img.requires_grad = True

            real_pred = discriminator(real_img)
            r1_loss = d_r1_loss(real_pred, real_img)

            discriminator.zero_grad()
            (args.gan_weight * (args.r1 / 2 * r1_loss * args.d_reg_every + 0 * real_pred[0])).backward()
            d_optim.step()
        
        loss_dict["r1"] = r1_loss

        # train generator
        requires_grad(generator, True)
        requires_grad(discriminator, False)

        if not args.lmdb:
            this_data = next(trainloader)
            real_img = this_data[0]
        else:
            real_img = next(trainloader)
            real_img = real_img.to(args.device)

        noise = torch.randn((args.batch_real, args.start_dim)).cuda()
        fake_img, _ = generator(noise)
        fake_pred = discriminator(fake_img)
        g_loss = g_nonsaturating_loss(fake_pred) * args.gan_weight

        loss_dict["g"] = g_loss
        generator.zero_grad()
        g_loss.backward()
        g_optim.step()

        accumulate(g_ema, g_module, accum)

        # finish one iteration and reduce loss dict
        loss_reduced = reduce_loss_dict(loss_dict)
        d_loss_val = loss_reduced["d"].mean().item()
        g_loss_val = loss_reduced["g"].mean().item()
        r1_val = loss_reduced["r1"].mean().item()

        if args.lr_decay and it > args.lr_decay_start_steps:
            args.G_lr -= lr_decay_per_step
            args.D_lr = args.G_lr * 4 if args.ttur else (args.D_lr - lr_decay_per_step)

            for param_group in d_optim.param_groups:
                param_group['lr'] = args.D_lr
            for param_group in g_optim.param_groups:
                param_group['lr'] = args.G_lr


        if get_rank() == 0:
            if it % args.log_freq == 0:
                log(d_loss_val, g_loss_val, r1_val, args.G_lr, args.D_lr, step=it)

            if it % args.print_freq == 0:
                iters_time = time.time() - end
                end = time.time()
                if args.lr_decay:
                    print("{} Iters: {}\tTime: {:.4f}\tD_loss: {:.4f}\tG_loss: {:.4f}\tR1: {:.4f}\tG_lr: {:e}\tD_lr: {:e}".format(get_time(), it, iters_time, d_loss_val, g_loss_val, r1_val, args.G_lr, args.D_lr))
                else: 
                    print("{} Iters: {}\tTime: {:.4f}\tD_loss: {:.4f}\tG_loss: {:.4f}\tR1: {:.4f}".format(get_time(), it, iters_time, d_loss_val, g_loss_val, r1_val))

            if it % args.save_freq == 0:
                save_checkpoint(g_module, d_module, g_ema, save_dir, step=it)

            if it != 0 and it % args.eval_freq == 0:
                save_checkpoint(g_module, d_module, g_ema, save_dir, step=it)

                print("=> Evaluation ...")
                g_ema.eval()
                fid_score = evaluation(g_ema, testloader, save_dir, args, step=it)

                wandb.log({'FID-score': fid_score}, step=it)
    
    wandb.finish()

def tensor_transform_reverse(image, mean, std):
    assert image.dim() == 4
    moco_input = torch.zeros(image.size()).type_as(image)
    moco_input[:,0,:,:] = image[:,0,:,:] * std[0] + mean[0]
    moco_input[:,1,:,:] = image[:,1,:,:] * std[1] + mean[1]
    moco_input[:,2,:,:] = image[:,2,:,:] * std[2] + mean[2]
    return moco_input    

def evaluation(generator, testloader, save_dir, args, step):
    eval_path = os.path.join(save_dir, "eval_{}".format(str(step)))
    os.makedirs(eval_path, exist_ok=False)

    cnt = 0
    for _ in tqdm(range(args.val_num_batches)):
        with torch.no_grad():
            noise = torch.randn((args.val_batch_size, args.start_dim)).cuda()

            out_sample, _ = generator(noise)
            out_sample = tensor_transform_reverse(out_sample, args.mean, args.std)

            for j in range(args.val_batch_size):
                torchvision.utils.save_image(
                    out_sample[j],
                    eval_path + f"/{str(cnt).zfill(6)}.png",
                    nrow=1,
                    padding=0,
                    normalize=True,
                    value_range=(0, 1),
                )
                cnt += 1
    
    device = torch.device('cuda:0')
    fid = fid_score.calculate_fid_given_path_testloader(eval_path, testloader, batch_size=args.val_batch_size, device=device, dims=2048)

    print("Fid Score : ({:.2f}, {:.1f}M)".format(fid, step / 1000000))

    return fid

if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument("--path", type=str, default="data", help="path of training data")
    parser.add_argument("--dataset", type=str, default="CIFAR10", help="which dataset to use")
    parser.add_argument("--subset", type=str, default="horse", help="which CIFAR10 subset to use", 
        choices=['', 'airplane', 'automobile', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck'])
    
    parser.add_argument("--iter", type=int, default=150000)
    parser.add_argument("--batch_real", type=int, default=4)

    parser.add_argument("--r1", type=float, default=5)
    parser.add_argument("--d_reg_every", type=int, default=16)

    parser.add_argument("--G_lr", type=float, default=0.0002)
    parser.add_argument("--D_lr", type=float, default=0.0002)
    parser.add_argument("--beta1", type=float, default=0.0)
    parser.add_argument("--beta2", type=float, default=0.99)

    parser.add_argument("--start_dim", type=int, default=256, help="Start dim of generator input dim")

    parser.add_argument("--D_channel_multiplier", type=int, default=2)
    parser.add_argument("--G_channel_multiplier", type=int, default=2)

    parser.add_argument("--local_rank", type=int, default=0)

    parser.add_argument("--print_freq", type=int, default=1000)
    parser.add_argument("--save_freq", type=int, default=1000)
    parser.add_argument("--eval_freq", type=int, default=10000)
    parser.add_argument("--log_freq", type=int, default=10)

    parser.add_argument('--workers', default=8, type=int, help='Number of workers')

    parser.add_argument('--val_num_batches', default=50, type=int, help='Num of batches will be generated during evalution')
    parser.add_argument('--val_batch_size', default=32, type=int, help='Batch size during evalution')

    parser.add_argument('--D_sn', action="store_true", help='If we use spectral norm in D')
    parser.add_argument('--ttur', action="store_true", help='If we use TTUR during training')
    parser.add_argument("--bcr", action="store_true", help='If we add bcr during training')

    parser.add_argument('--gan_weight', default=1, type=float, help='Gan loss weight')
    parser.add_argument('--mlp_ratio', default=4, type=int, help='MLP ratio in swin')
    parser.add_argument("--lr_mlp", default=0.01, type=float, help='Lr mul for 8 * fc')

    parser.add_argument("--bcr_fake_lambda", default=10, type=float, help='Bcr weight for fake data')
    parser.add_argument("--bcr_real_lambda", default=10, type=float, help='Bcr weight for real data')

    parser.add_argument("--enable_full_resolution", default=8, type=int, help='Enable full resolution attention index')

    parser.add_argument("--lr_decay", action="store_true", help='Whether to use lr decay')
    parser.add_argument("--lr_decay_start_steps", default=100000, type=int, help='Steps to start lr decay')

    parser.add_argument("--lmdb", action="store_true", help='Whether to use lmdb datasets')
    

    args, _ = parser.parse_known_args()

    args.device = "cuda" if torch.cuda.is_available() else "cpu"
    n_gpu = torch.cuda.device_count()
    args.distributed = n_gpu > 1

    args.D_sn = True
    args.ttur = True
    args.lr_decay = True

    args.n_mlp = 4
    args.g_reg_every = 1e7 # We do not apply regularization on g

    if args.distributed:
        torch.cuda.set_device(args.local_rank)
        torch.distributed.init_process_group(backend="nccl", init_method="env://", timeout=timedelta(0, 18000))
        synchronize()

    if args.distributed and get_rank() != 0:
        def print_pass(*args):
            pass
        builtins.print = print_pass

    channel, im_size, _, _, mean, std, dst_train, testloader = get_dataset(args)

    args.size, args.mean, args.std = im_size, mean, std

    trainloader = torch.utils.data.DataLoader(dst_train, batch_size=args.batch_real, shuffle=True)

    generator = Generator(args.size, args.start_dim, args.n_mlp, channel_multiplier=args.G_channel_multiplier, lr_mlp=args.lr_mlp, enable_full_resolution=args.enable_full_resolution).to(args.device)
    g_ema = Generator(args.size, args.start_dim, args.n_mlp, channel_multiplier=args.G_channel_multiplier, lr_mlp=args.lr_mlp, enable_full_resolution=args.enable_full_resolution).to(args.device)
    discriminator = Discriminator(args.size, channel_multiplier=args.D_channel_multiplier, sn=args.D_sn).to(args.device)
    accumulate(g_ema, generator, 0)

    g_reg_ratio = args.g_reg_every / (args.g_reg_every + 1)
    d_reg_ratio = args.d_reg_every / (args.d_reg_every + 1)

    print("-" * 80)
    print("Generator: ")
    print(generator)
    print("-" * 80)
    print("Discriminator: ")
    print(discriminator)

    if args.distributed:
        generator = nn.parallel.DistributedDataParallel(
            generator,
            device_ids=[args.local_rank],
            output_device=args.local_rank,
            broadcast_buffers=False,
        )

        discriminator = nn.parallel.DistributedDataParallel(
            discriminator,
            device_ids=[args.local_rank],
            output_device=args.local_rank,
            broadcast_buffers=False,
        )

    g_optim = optim.Adam(
        generator.parameters(),
        lr=args.G_lr * g_reg_ratio if not args.ttur else args.D_lr / 4 * g_reg_ratio,
        betas=(args.beta1 ** g_reg_ratio, args.beta2 ** g_reg_ratio),
    )
    d_optim = optim.Adam(
        discriminator.parameters(),
        lr=args.D_lr * d_reg_ratio,
        betas=(args.beta1 ** d_reg_ratio, args.beta2 ** d_reg_ratio),
    )

    train(trainloader, testloader, generator, discriminator, g_optim, d_optim, g_ema, args)

