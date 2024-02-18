import argparse
import os
from functools import partial
import math

import torch
import torch.distributed as dist
import yaml
from datasets import get_dataset
from metric import KNN, LinearProbe
from torchvision.utils import make_grid, save_image
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm
from ema_pytorch import EMA

from model.models import get_models_class
from utils import Config, get_optimizer, init_seeds, reduce_tensor, DataLoaderDDP, print0, Meter

# ===== training =====

def train(opt):
    yaml_path = opt.config
    local_rank = opt.local_rank
    use_amp = opt.use_amp

    with open(yaml_path, 'r') as f:
        opt = yaml.full_load(f)
    print0(opt)
    opt = Config(opt)
    model_dir = os.path.join(opt.save_dir, "ckpts")
    vis_dir = os.path.join(opt.save_dir, "visual")
    tsbd_dir = os.path.join(opt.save_dir, "tensorboard")
    if local_rank == 0:
        os.makedirs(model_dir, exist_ok=True)
        os.makedirs(vis_dir, exist_ok=True)

    device = "cuda:%d" % local_rank
    DIFFUSION, NETWORK = get_models_class(opt.model_type, opt.net_type)
    diff = DIFFUSION(nn_model=NETWORK(**opt.network),
                     **opt.diffusion,
                     device=device,
                     )
    diff.to(device)
    if local_rank == 0:
        ema = EMA(diff, beta=opt.ema, update_after_step=0, update_every=1)
        ema.to(device)
        ema.eval()
        writer = SummaryWriter(log_dir=tsbd_dir)

    diff = torch.nn.SyncBatchNorm.convert_sync_batchnorm(diff)
    diff = torch.nn.parallel.DistributedDataParallel(
        diff, device_ids=[local_rank], output_device=local_rank)

    train_set = get_dataset(name=opt.dataset, root="./data", train=True, flip=opt.flip)
    print0("train dataset:", len(train_set))

    if local_rank == 0:
        down_train = get_dataset(name=opt.dataset, root="./data", train=True, flip=True)
        down_test = get_dataset(name=opt.dataset, root="./data", train=False)
        knn = KNN(down_train, down_test, opt.linear['batch_size'])
        lp = LinearProbe(down_train, down_test, opt.classes, opt.linear['batch_size'], opt.linear['lrate'], opt.linear['n_epoch'])

    train_loader, sampler = DataLoaderDDP(train_set,
                                          batch_size=opt.batch_size,
                                          shuffle=True)

    lr = opt.lrate
    DDP_multiplier = dist.get_world_size()
    print0("Using DDP, lr = %f * %d" % (lr, DDP_multiplier))
    lr *= DDP_multiplier
    optim = get_optimizer(diff.parameters(), opt, lr=lr)
    scaler = torch.cuda.amp.GradScaler(enabled=use_amp)

    if opt.load_epoch != -1:
        target = os.path.join(model_dir, f"model_{opt.load_epoch}.pth")
        print0("loading model at", target)
        checkpoint = torch.load(target, map_location=device)
        diff.load_state_dict(checkpoint['MODEL'])
        if local_rank == 0:
            ema.load_state_dict(checkpoint['EMA'])
        optim.load_state_dict(checkpoint['opt'])

    for ep in range(opt.load_epoch + 1, opt.n_epoch):
        for g in optim.param_groups:
            if ep < opt.warm_epoch:
                g['lr'] = lr * min((ep + 1.0) / opt.warm_epoch, 1.0) # warmup
            else:
                g['lr'] = lr / math.sqrt(max(ep / opt.tref_epoch, 1.0)) # inverse square root
        sampler.set_epoch(ep)
        dist.barrier()
        # training
        diff.train()
        if local_rank == 0:
            now_lr = optim.param_groups[0]['lr']
            print(f'epoch {ep}, lr {now_lr:f}')
            meter = Meter(n_items=1)
            pbar = tqdm(train_loader)
        else:
            pbar = train_loader
        for x, c in pbar:
            optim.zero_grad()
            x = x.to(device)
            loss = diff(x, use_amp=use_amp)
            scaler.scale(loss).backward()
            scaler.step(optim)
            scaler.update()

            # logging
            dist.barrier()
            loss = reduce_tensor(loss)
            if local_rank == 0:
                ema.update()
                meter.update([loss.item()])
                pbar.set_description(f"loss: {meter.get(0):.4f}")

        # testing
        if local_rank == 0:
            writer.add_scalar('lr', now_lr, ep)
            writer.add_scalar('loss', meter.get(0), ep)
            if ep % 100 == 0 or ep == opt.n_epoch - 1:
                pass
            else:
                continue

            if ep > 0:
                print(f'epoch {ep}, evaluating:')
                feat_func = partial(ema.ema_model.get_feature, t=opt.linear['timestep'], name=opt.linear['blockname'], norm=False, use_amp=use_amp)
                test_knn = knn.evaluate(feat_func)
                test_lp = lp.evaluate(feat_func)
                writer.add_scalar('metrics/K Nearest Neighbors', test_knn, ep)
                writer.add_scalar('metrics/Linear Probe', test_lp, ep)

            if opt.model_type == 'EDM':
                ema_sample_method = ema.ema_model.edm_sample
            else:
                raise NotImplementedError

            ema.ema_model.eval()
            with torch.no_grad():
                x_gen = ema_sample_method(opt.n_sample, x.shape[1:])
            # save an image of currently generated samples (top rows)
            # followed by real images (bottom rows)
            x_real = x[:opt.n_sample]
            x_all = torch.cat([x_gen.cpu(), x_real.cpu()])
            grid = make_grid(x_all, nrow=10)

            save_path = os.path.join(vis_dir, f"image_ep{ep}_ema.png")
            save_image(grid, save_path)
            print('saved image at', save_path)

            # optionally save model
            if opt.save_model:
                checkpoint = {
                    'MODEL': diff.state_dict(),
                    'EMA': ema.state_dict(),
                    'opt': optim.state_dict(),
                }
                save_path = os.path.join(model_dir, f"model_{ep}.pth")
                torch.save(checkpoint, save_path)
                print('saved model at', save_path)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str)
    parser.add_argument("--use_amp", action='store_true', default=False)
    opt = parser.parse_args()
    opt.local_rank = int(os.environ['LOCAL_RANK'])
    print0(opt)

    init_seeds(no=opt.local_rank)
    dist.init_process_group(backend='nccl')
    torch.cuda.set_device(opt.local_rank)
    train(opt)
