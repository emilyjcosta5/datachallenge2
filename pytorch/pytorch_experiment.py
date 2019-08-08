from __future__ import print_function

import os
import h5py
import torch
import argparse
import torch.backends.cudnn as cudnn
import torch.nn.functional as F
import torch.optim as optim
import torch.utils.data.distributed
import torch.distributed as dist
import horovod.torch as hvd
import tensorboardX
import math
import subprocess
from tqdm import tqdm
from torch.utils.data import Dataset, DataLoader
from torchvision import models


# Training settings
parser = argparse.ArgumentParser(description='Smoky Mountain Data Challenge in PyTorch',
                                 formatter_class=argparse.ArgumentDefaultsHelpFormatter)
parser.add_argument('--train_dir', default=os.path.expanduser('~/imagenet/train'),
                    help='path to training data')
parser.add_argument('--val_dir', default=os.path.expanduser('~/imagenet/validation'),
                    help='path to validation data')
parser.add_argument('--batch_size', type=int, default=32,
                    help='input batch size for training')
parser.add_argument('--val_batch_size', type=int, default=32,
                    help='input batch size for validation')
parser.add_argument('--log_dir', default='./logs',
                    help='tensorboard log directory')
parser.add_argument('--epochs', type=int, default=90,
                    help='number of epochs to train')
parser.add_argument('--seed', type=int, default=42,
                    help='random seed')
parser.add_argument('--wd', type=float, default=0.00005,
                    help='weight decay')
parser.add_argument('--momentum', type=float, default=0.9,
                    help='SGD momentum')
parser.add_argument('--batches-per-allreduce', type=int, default=1,
                    help='number of batches processed locally before '
                    'executing allreduce across workers; it multiplies '
                    'total batch size.')
parser.add_argument('--verbose', type=bool, default=False,
                    help='turns on data load progress update on STDOUT')
args = parser.parse_args()


class HDF5Dataset(Dataset):
    """
    Custom PyTorch Dataset from HDF5 files for SMC
    args:
        root_dir: a path for the directory that contains .h5 files
        data_files: a list of filenames under the root_dir
    """
    def __init__(self, root_dir, data_files):
        full_data_files = [os.path.join(root_dir, file) for file in data_files]
        self.data_files = []
        # self.data_files contains tuples of filename and key pairs
        # This pair can uniquely identify one file
        for filename in full_data_files:
            f = h5py.File(filename, "r")
            for key in f.keys():
                self.data_files.append((filename, key))
            f.close()

    def __getitem__(self, index):
        return self._load_hdf5_file(index)

    def __len__(self):
        return len(self.data_files)
    
    def _load_hdf5_file(self, index):
        if index % 1000 == 0 and verbose:
            print("Processing {}th file!".format(index))
        hdf5_file, key = self.data_files[index]
        f = h5py.File(hdf5_file, "r")
        sample = f[key]
        y = int(sample.attrs['space_group'])
        x = sample['cbed_stack'].value

        return (x, y)


def main_experiment():
    torch.manual_seed(args.seed)
    
    # Distributed training setup (Doesn't work yet lol)
    '''
    import socket
    hostname = socket.gethostname() 
    IP = socket.gethostbyname(hostname) 
    os.environ["MASTER_ADDR"] = IP

    get_cnodes = "echo $(cat {} | sort | uniq | grep -v batch | grep -v login)".format(os.environ['LSB_DJOB_HOSTFILE'])
    cnodes = subprocess.check_output(get_cnodes, shell=True)
    cnodes = str(cnodes)[2:-3].split(' ')
    nnodes = len(cnodes)
    rank = int(os.environ['PMIX_RANK']) 
    print("RANK:", rank)
    dist.init_process_group(backend="nccl",
                            world_size=nnodes, 
                            rank=rank)
    
    #dist.init_process_group(backend="nccl")
    '''

    print("======= START LOADING DATA =========")
    #train_n = len(os.listdir(args.train_dir))
    num_files = len(os.listdir(args.train_dir))
    train_n = np.floor(num_files * 0.8)
    train_files = sorted(os.listdir(args.train_dir))[:train_n]
    train_dataset = HDF5Dataset(args.train_dir, train_files)
    
    val_files = sorted(os.listdir(args.train_dir))[train_n:]
    val_dataset = HDF5Dataset(args.val_dir, val_files)
    print("Training size")
    print(len(train_dataset))
    print("Dev size")
    print("Test size")
    print(len(val_dataset))

    train_loader = DataLoader(dataset=train_dataset,
                              batch_size=args.batch_size,
                              shuffle=True,
                              num_workers=2)

    val_loader = DataLoader(dataset=val_dataset, 
                            batch_size=args.batch_size, 
                            shuffle=True)

    # Set up standard ResNet-50 model.
    model = models.resnet50()
    model.cuda()
#     model = torch.nn.parallel.DistributedDataParallel(model)

    # Horovod: scale learning rate by the number of GPUs.
    # Gradient Accumulation: scale learning rate by batches_per_allreduce
    optimizer = optim.SGD(model.parameters(), lr=0.01, 
                          momentum=args.momentum, weight_decay=args.wd)
    
    log_writer = tensorboardX.SummaryWriter(args.log_dir)

    def train(epoch):
        model.train()
        train_loss = Metric('train_loss')
        train_accuracy = Metric('train_accuracy')

        with tqdm(total=len(train_loader),
                  desc='Train Epoch     #{}'.format(epoch + 1)) as t:
            for batch_idx, (data, target) in enumerate(train_loader):
                data, target = data.cuda(), target.cuda()
                optimizer.zero_grad()
                # Split data into sub-batches of size batch_size
                for i in range(0, len(data), args.batch_size):
                    data_batch = data[i:i + args.batch_size]
                    target_batch = target[i:i + args.batch_size]
                    output = model(data_batch)
                    train_accuracy.update(accuracy(output, target_batch))
                    loss = F.cross_entropy(output, target_batch)
                    train_loss.update(loss)
                    # Average gradients among sub-batches
                    loss.div_(math.ceil(float(len(data)) / args.batch_size))
                    loss.backward()
                # Gradient is applied across all ranks
                optimizer.step()
                t.set_postfix({'loss': train_loss.avg.item(),
                               'accuracy': 100. * train_accuracy.avg.item()})
                t.update(1)

        if log_writer:
            log_writer.add_scalar('train/loss', train_loss.avg, epoch)
            log_writer.add_scalar('train/accuracy', train_accuracy.avg, epoch)


    def validate(epoch):
        model.eval()
        val_loss = Metric('val_loss')
        val_accuracy = Metric('val_accuracy')

        with tqdm(total=len(val_loader),
                  desc='Validate Epoch  #{}'.format(epoch + 1)) as t:
            with torch.no_grad():
                for data, target in val_loader:
                    data, target = data.cuda(), target.cuda()
                    output = model(data)

                    val_loss.update(F.cross_entropy(output, target))
                    val_accuracy.update(accuracy(output, target))
                    t.set_postfix({'loss': val_loss.avg.item(),
                                   'accuracy': 100. * val_accuracy.avg.item()})
                    t.update(1)

        if log_writer:
            log_writer.add_scalar('val/loss', val_loss.avg, epoch)
            log_writer.add_scalar('val/accuracy', val_accuracy.avg, epoch)
    
    def accuracy(output, target):
        # get the index of the max log-probability
        pred = output.max(1, keepdim=True)[1]
        return pred.eq(target.view_as(pred)).cpu().float().mean()
        

    # Horovod: average metrics from distributed training.
    class Metric(object):
        def __init__(self, name):
            self.name = name
            self.sum = torch.tensor(0.)
            self.n = torch.tensor(0.)

        def update(self, val):
            self.sum += val
            self.n += 1

        @property
        def avg(self):
            return self.sum / self.n


    for epoch in range(args.epochs):
        train(epoch)
        validate(epoch)


 

def main():
   
    hvd.init()
    torch.manual_seed(args.seed)
    # Horovod: print logs on the first worker.
    verbose = 1 if hvd.rank() == 0 else 0

    # Horovod: write TensorBoard logs on first worker.
    log_writer = tensorboardX.SummaryWriter(args.log_dir) if hvd.rank() == 0 else None
    
    kwargs = {'num_workers': 0, 'pin_memory': True}

    print("======= START LOADING DATA =========")
    #train_n = len(os.listdir(args.train_dir))
    train_n = 30
    train_files = sorted(os.listdir(args.train_dir))[:train_n]
    #train_files = ["batch_train_{}.h5".format(i) for i in range(train_n)]
    #train_files = ["batch_train_0.h5", "batch_train_1.h5"] 
    train_dataset = HDF5Dataset(args.train_dir, train_files)
    '''
    train_loader = DataLoader(dataset=train_dataset,
                              batch_size=args.batch_size,
                              shuffle=True,
                              num_workers=2)
    '''
    
    val_n = 10
    val_files = sorted(os.listdir(args.val_dir))[:val_n]
    val_dataset = HDF5Dataset(args.val_dir, val_files)
    '''
    val_loader = DataLoader(dataset=val_dataset, 
                            batch_size=args.batch_size, 
                            shuffle=True)
    '''
    print("Training size")
    print(len(train_dataset))
    print("Dev size")
    print("Test size")
    print(len(val_dataset))
   
    # Horovod: use DistributedSampler to partition data among workers. Manually specify
    # `num_replicas=hvd.size()` and `rank=hvd.rank()`.
    train_sampler = torch.utils.data.distributed.DistributedSampler(
            train_dataset, num_replicas=hvd.size(), rank=hvd.rank())
    train_loader = torch.utils.data.DataLoader(
            train_dataset, batch_size=args.batch_size,
            sampler=train_sampler, **kwargs)
    val_sampler = torch.utils.data.distributed.DistributedSampler(
            val_dataset, num_replicas=hvd.size(), rank=hvd.rank())
    val_loader = torch.utils.data.DataLoader(
            val_dataset, batch_size=args.val_batch_size,
            sampler=val_sampler, **kwargs)

    # Set up standard ResNet-50 model.
    model = models.resnet50()
    model.cuda()

    # Horovod: scale learning rate by the number of GPUs.
    # Gradient Accumulation: scale learning rate by batches_per_allreduce
    optimizer = optim.SGD(model.parameters(), lr=0.01, 
                          momentum=args.momentum, weight_decay=args.wd)
    compression = hvd.Compression.fp16

    # Horovod: wrap optimizer with DistributedOptimizer.
    optimizer = hvd.DistributedOptimizer(
                optimizer, named_parameters=model.named_parameters(),
                compression=compression,
                backward_passes_per_step=args.batches_per_allreduce)

    # Horovod: broadcast parameters & optimizer state.
    hvd.broadcast_parameters(model.state_dict(), root_rank=0)
    hvd.broadcast_optimizer_state(optimizer, root_rank=0)

    def train(epoch):
        model.train()
        train_sampler.set_epoch(epoch)
        train_loss = Metric('train_loss')
        train_accuracy = Metric('train_accuracy')

        with tqdm(total=len(train_loader),
                  desc='Train Epoch     #{}'.format(epoch + 1),
                  disable=not verbose) as t:
            for batch_idx, (data, target) in enumerate(train_loader):
                # print(data, target)
                if args.cuda:
                    data, target = data.cuda(), target.cuda()
                optimizer.zero_grad()
                # Split data into sub-batches of size batch_size
                for i in range(0, len(data), args.batch_size):
                    data_batch = data[i:i + args.batch_size]
                    target_batch = target[i:i + args.batch_size]
                    output = model(data_batch)
                    train_accuracy.update(accuracy(output, target_batch))
                    loss = F.cross_entropy(output, target_batch)
                    train_loss.update(loss)
                    # Average gradients among sub-batches
                    loss.div_(math.ceil(float(len(data)) / args.batch_size))
                    loss.backward()
                # Gradient is applied across all ranks
                optimizer.step()
                t.set_postfix({'loss': train_loss.avg.item(),
                               'accuracy': 100. * train_accuracy.avg.item()})
            t.update(1)

        if log_writer:
            log_writer.add_scalar('train/loss', train_loss.avg, epoch)
            log_writer.add_scalar('train/accuracy', train_accuracy.avg, epoch)


    def validate(epoch):
        model.eval()
        val_loss = Metric('val_loss')
        val_accuracy = Metric('val_accuracy')

        with tqdm(total=len(val_loader),
                  desc='Validate Epoch  #{}'.format(epoch + 1),
                  disable=not verbose) as t:
            with torch.no_grad():
                for data, target in val_loader:
                    if args.cuda:
                        data, target = data.cuda(), target.cuda()
                    output = model(data)

                    val_loss.update(F.cross_entropy(output, target))
                    val_accuracy.update(accuracy(output, target))
                    t.set_postfix({'loss': val_loss.avg.item(),
                                   'accuracy': 100. * val_accuracy.avg.item()})
                    t.update(1)

        if log_writer:
            log_writer.add_scalar('val/loss', val_loss.avg, epoch)
            log_writer.add_scalar('val/accuracy', val_accuracy.avg, epoch)
    
    def accuracy(output, target):
        # get the index of the max log-probability
        pred = output.max(1, keepdim=True)[1]
        return pred.eq(target.view_as(pred)).cpu().float().mean()
        

    # Horovod: average metrics from distributed training.
    class Metric(object):
        def __init__(self, name):
            self.name = name
            self.sum = torch.tensor(0.)
            self.n = torch.tensor(0.)

        def update(self, val):
            self.sum += hvd.allreduce(val.detach().cpu(), name=self.name)
            self.n += 1

        @property
        def avg(self):
            return self.sum / self.n


    for epoch in range(resume_from_epoch, args.epochs):
        train(epoch)
        validate(epoch)


if __name__ == "__main__":
    main_experiment()
