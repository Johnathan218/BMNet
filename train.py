import torch
import numpy as np
import os
import time
import datetime
import shutil
from tqdm import tqdm
from argparse import ArgumentParser
import tensorboardX
import torch.nn as nn
import torch.distributed as dist
import torch.backends.cudnn as cudnn
from PIL import Image
from torchvision import transforms
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler
from apex import amp
from apex.parallel import DistributedDataParallel as DDP

from data.dotadataset import make_dataset
from utils import batch_PSNR, time2file_name, AverageMeter, measurement_vmax, quantize_measurement
from network.BMNet import BMNet
from timm.scheduler import CosineLRScheduler
import einops

cudnn.benchmark = True

parser = ArgumentParser(description='BMI')

parser.add_argument('--warmup_steps', type=int, default=5, help='epoch number of learnig rate warmup')
parser.add_argument('--start_epoch', type=int, default=0, help='epoch number of start training')
parser.add_argument('--end_epoch', type=int, default=100, help='epoch number of end training')
parser.add_argument('--learning_rate', type=float, default=5e-6, help='learning rate')
parser.add_argument('--batch_size', type=int, default=1, help='batch size')
parser.add_argument('--image_size', type=int, nargs='+', default=[512, 512], help='image size')
parser.add_argument('--resize_size', type=int, nargs='+', default=None, help='resize the image to fit the compression ratio')
parser.add_argument('--n_channels', type=int, default=1, help='1 for gray image, currently only support 1')
parser.add_argument("--cs_ratio", type=int, nargs='+', default=[4, 4], help="compression ratio")
parser.add_argument('--num_stage', type=int, default=10, help='satge number of the DUN')
parser.add_argument('--scaler', action='store_true', help='whether add scaler to mask')
parser.add_argument('--lm', action='store_true', help='whether learnable mask')
parser.add_argument('--use_checkpoint', action='store_true', help='whether use torch checkpoint to save memory')

parser.add_argument('--save_interval', type=int, default=5, help='save model on test set every x epochs')
parser.add_argument("--test_interval", type=int, default=1, help='evaluate model on test set every x epochs')
parser.add_argument("--data_path", type=str, default="/data2/wangzhibin/DOTA/trainsplit512_nogap/images/", help='path to dota dataset')
parser.add_argument('--save_dir', type=str, default='./model_ckpt', help='output dir')
parser.add_argument('--finetune', type=str, default=None, help='resume path')
parser.add_argument('--resume', type=str, default=None, help='resume path')

parser.add_argument('--quant_meas', action='store_true', help='quantize the measurement before decoding')
parser.add_argument('--quant_bits', type=int, default=8, help='bit depth of the quantized measurement')
parser.add_argument('--quant_range', type=str, default='global', choices=['global', 'phisum'],
                    help="quantization range: 'global' uses [0, N], 'phisum' uses the per-position bound")

parser.add_argument('--opt-level', type=str, default='O0', help='use fp32 or fp16')
parser.add_argument("--local_rank", default=0, type=int)
parser.add_argument("--seed", default=42, type=int)
args = parser.parse_args()

start_epoch = args.start_epoch
end_epoch = args.end_epoch
batch_size = args.batch_size
best_psnr = 0.

args.local_rank = int(os.getenv('LOCAL_RANK', -1))  # https://pytorch.org/docs/stable/elastic/run.html
args.world_size = int(os.getenv('WORLD_SIZE', 1))

if args.local_rank != -1:
    torch.cuda.set_device(args.local_rank)
    torch.distributed.init_process_group(backend='nccl', init_method='env://')

# scale learning_rate according to total_batch_size
total_batch_size = args.world_size * args.batch_size
args.learning_rate = args.learning_rate * float(total_batch_size)
args.init_learning_rate = args.learning_rate / 5.
args.min_learning_rate = args.learning_rate / 100.

cr1, cr2 = args.cs_ratio


g = torch.Generator()
g.manual_seed(args.seed)

mask = torch.bernoulli(torch.empty(cr1 * cr2, 1, args.image_size[0] // cr1, args.image_size[1] // cr2).uniform_(0, 1), generator=g)
if args.resize_size:
    mask = torch.bernoulli(torch.empty(cr1 * cr2, 1, args.resize_size[0] // cr1, args.resize_size[1] // cr2).uniform_(0, 1), generator=g)


def main():
    train_dir = args.data_path
    test_dir = train_dir.replace('train', 'val')
    train_dataset, test_dataset = make_dataset(train_dir, test_dir, cr=max(args.cs_ratio), seed=args.seed)

    if args.local_rank != -1:
        sampler = DistributedSampler(train_dataset, num_replicas=args.world_size, rank=args.local_rank, shuffle=True)
        train_dataloader = DataLoader(train_dataset, batch_size=args.batch_size, sampler=sampler, pin_memory=True, num_workers=8)
    else:
        train_dataloader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, pin_memory=True,
                                      num_workers=8)
    test_dataloader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False, num_workers=8)

    # model defination
    model = BMNet(
        in_chans=args.n_channels,
        embed_dim=32,
        num_stage=args.num_stage,
        cs_ratio=args.cs_ratio,
        scaler=args.scaler,
        use_checkpoint=args.use_checkpoint
    ).cuda()

    # whether learnable mask
    model.mask = nn.Parameter(mask.cuda(), requires_grad=args.lm)

    if args.finetune is not None:
        checkpoint_filename = args.finetune
        if os.path.isfile(checkpoint_filename):
            print("=> loading weights '{}'".format(checkpoint_filename))
            checkpoint = torch.load(checkpoint_filename, map_location='cpu')
            checkpoint['state_dict'] = {k.replace('module.', ''): v for k, v in checkpoint['state_dict'].items()}
            del checkpoint['mask']
            if 'mask' in checkpoint['state_dict']:
                del checkpoint['state_dict']['mask']
            model.load_state_dict(checkpoint['state_dict'], strict=False)
            print("=> loaded weights '{}' (epoch {})".format(checkpoint_filename, checkpoint['epoch']))
        else:
            print("=> no model weights found at '{}'".format(checkpoint_filename))

    # loss and optimizer
    criterion = nn.MSELoss().cuda()
    optimizer = torch.optim.Adam(model.parameters(), lr=args.learning_rate)
    if args.local_rank in [-1, 0]:
        tot_grad_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        print('number of params (M): %.2f' % (tot_grad_params / 1.e6))

    # lr scheduler
    scheduler = CosineLRScheduler(
        optimizer=optimizer,
        t_initial=args.end_epoch,
        lr_min=args.min_learning_rate,
        warmup_t=args.warmup_steps,
        warmup_lr_init=args.init_learning_rate
    )

    model, optimizer = amp.initialize(model, optimizer, opt_level=args.opt_level)

    if args.local_rank != -1:
        model = DDP(model, delay_allreduce=True)

    global save_dir
    # load model and make model saving/log dir
    date_time = str(datetime.datetime.now())
    date_time = time2file_name(date_time)
    save_dir = args.save_dir + '/' + date_time

    if not os.path.exists(save_dir) and args.local_rank in [-1, 0]:
        os.makedirs(save_dir)
        # tensorboardX
        writer = tensorboardX.SummaryWriter(log_dir=save_dir)

    if args.resume is not None:
        # Use a local scope to avoid dangling referencesa
        def resume():
            checkpoint_filename = args.resume
            if os.path.isfile(checkpoint_filename):
                print("=> loading checkpoint '{}'".format(checkpoint_filename))
                checkpoint = torch.load(checkpoint_filename, map_location='cpu')
                if not isinstance(model, DDP):
                    checkpoint['state_dict'] = {k.replace('module.', ''): v for k, v in checkpoint['state_dict'].items()}
                global best_psnr
                best_psnr = checkpoint['best_psnr']
                model.load_state_dict(checkpoint['state_dict'], strict=False)
                optimizer.load_state_dict(checkpoint['optimizer'])
                amp.load_state_dict(checkpoint['amp'])
                print("=> loaded checkpoint '{}' (epoch {})".format(checkpoint_filename, checkpoint['epoch']))
            else:
                print("=> no checkpoint found at '{}'".format(checkpoint_filename))

        resume()

    # Training
    iter = 0

    for epoch_i in range(start_epoch + 1, end_epoch + 1):
        scheduler.step(epoch_i - 1)
        if args.local_rank in [-1, 0]:
            lr = optimizer.param_groups[0]['lr']
            print("current learning rate: %e" % lr)
            writer.add_scalar('lr', lr, epoch_i)
            time_start = time.time()
        if args.local_rank != -1:
            sampler.set_epoch(epoch_i)
        for i, data in enumerate(train_dataloader):
            # break
            iter += 1

            model.train()
            model.zero_grad()
            optimizer.zero_grad()

            bs = data.shape[0]

            img_train = data.type(torch.FloatTensor).cuda()

            if args.resize_size:
                t = transforms.Resize(args.resize_size)
                input_img = t(img_train)
            else:
                input_img = img_train

            input_img = einops.rearrange(input_img, "b c (cr1 h) (cr2 w) -> b (cr1 cr2) c h w", cr1=cr1, cr2=cr2)

            if isinstance(model, DDP):
                input_mask = model.module.mask.unsqueeze(0).expand(bs, -1, -1, -1, -1) * model.module.scaler
            else:
                input_mask = model.mask.unsqueeze(0).expand(bs, -1, -1, -1, -1) * model.scaler

            meas = torch.sum(input_img * input_mask, dim=1, keepdim=True)

            if args.quant_meas:
                vmax = measurement_vmax(input_mask, args.quant_range, cr1 * cr2)
                meas = quantize_measurement(meas, args.quant_bits, vmax)

            out_train = model(meas, input_mask)

            if args.resize_size:
                t = transforms.Resize(args.image_size)
                out_train = t(out_train)

            loss = criterion(out_train, img_train)

            with amp.scale_loss(loss, optimizer) as scaled_loss:
                scaled_loss.backward()

            optimizer.step()

            # results
            out_train = torch.clamp(out_train, 0., 1.)
            psnr_train = batch_PSNR(out_train, img_train, 1.)
            psnr_train = torch.as_tensor(float(psnr_train)).cuda()
            if args.local_rank != -1:
                loss = reduce_tensor(loss)
                psnr_train = reduce_tensor(psnr_train)
            if args.local_rank in [-1, 0]:
                writer.add_scalar('psnr', psnr_train.item(), iter)
                writer.add_scalar('recon_loss', loss.item(), iter)
                print("%s [epoch %d][%d/%d] recon_loss: %.4f PSNR_train: %.2f dB" %
                      (time.strftime("%Y-%m-%d %H:%M:%S", time.localtime()), epoch_i, i, len(train_dataloader),
                       loss.item(), psnr_train.item()))
                # print(model.module.mask[0][0][0][:5])
            # break
        if args.local_rank in [-1, 0]:
            time_end = time.time()
        if args.local_rank in [-1, 0]:
            print('time cost', time_end - time_start)

        if args.local_rank in [-1, 0]:
            # evaluation
            psnr_avg_meter = AverageMeter()
            model.eval()
            show_test = 5
            print('#################Testing##################')
            for data in tqdm(test_dataloader):
                bs = data.shape[0]
                img_test = data.type(torch.FloatTensor).cuda()

                if args.resize_size:
                    t = transforms.Resize(args.resize_size)
                    input_img = t(img_test)
                else:
                    input_img = img_test

                if show_test:
                    test = data[0, :]
                    writer.add_image(f'img_{show_test}', test)

                input_img = einops.rearrange(input_img, "b c (cr1 h) (cr2 w) -> b (cr1 cr2) c h w", cr1=cr1, cr2=cr2)
                if isinstance(model, DDP):
                    input_mask = model.module.mask.unsqueeze(0).expand(bs, -1, -1, -1, -1) * model.module.scaler
                else:
                    input_mask = model.mask.unsqueeze(0).expand(bs, -1, -1, -1, -1) * model.scaler
                meas = torch.sum(input_img * input_mask, dim=1, keepdim=True)

                if args.quant_meas:
                    vmax = measurement_vmax(input_mask, args.quant_range, cr1 * cr2)
                    meas = quantize_measurement(meas, args.quant_bits, vmax)

                if show_test:
                    test = meas[0, :]
                    writer.add_images(f'measurement_{show_test}', test)

                with torch.no_grad():
                    out_test = model(meas, input_mask)
                    if args.resize_size:
                        t = transforms.Resize(args.image_size)
                        out_test = t(out_test)

                out_test = torch.clamp(out_test, 0, 1)

                if show_test:
                    test = out_test[0, :]
                    writer.add_image(f'reconstruction_{show_test}', test, epoch_i)
                    show_test = show_test - 1

                psnr_test = batch_PSNR(out_test, img_test, 1.)
                psnr_avg_meter.update(psnr_test)

            writer.add_scalar("avg", psnr_avg_meter.avg.item(), epoch_i)
            print("test psnr: %.4f" % psnr_avg_meter.avg.item())

            if args.local_rank in [-1, 0] and epoch_i % args.save_interval == 0:
                global best_psnr
                is_best = psnr_avg_meter.avg > best_psnr
                best_psnr = max(psnr_avg_meter.avg, best_psnr)
                save_checkpoint({
                    'epoch': epoch_i,
                    'state_dict': model.state_dict(),
                    'best_psnr': best_psnr,
                    'optimizer': optimizer.state_dict(),
                    'amp': amp.state_dict(),
                    'mask': model.module.mask if isinstance(model, DDP) else model.mask
                }, is_best, filename=os.path.join(save_dir, 'net_%d.pth' % epoch_i))
                print('best test psnr till now %.4f' % best_psnr)
                print('checkpoint with %d iterations has been saved.' % epoch_i)
                print()
                print()
            psnr_avg_meter.reset()


def save_checkpoint(state, is_best, filename='checkpoint.pth'):
    torch.save(state, filename)
    if is_best:
        shutil.copyfile(filename, os.path.join(save_dir, 'model_best.pth'))


def reduce_tensor(tensor):
    rt = tensor.clone()
    dist.all_reduce(rt, op=dist.ReduceOp.SUM)
    rt /= args.world_size
    return rt


if __name__ == "__main__":
    main()
