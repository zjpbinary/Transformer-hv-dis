from makemodel import *
from lodaBPEdata import *
from modules.labelsmoothing import *
from trainer import *
from eval import *
from torch.optim.lr_scheduler import ExponentialLR
import argparse
import torch.distributed as dist

parser = argparse.ArgumentParser()
parser.add_argument('--local_rank', default=0, type=int, help='node rank for distributed training')
args = parser.parse_args()

dist.init_process_group(backend='nccl')
torch.cuda.set_device(args.local_rank)

train_sampler = torch.utils.data.distributed.DistributedSampler(trainset)
#test_sampler = torch.utils.data.distributed.DistributedSampler(testset)
#val_sampler = torch.utils.data.distributed.DistributedSampler(valset)

train_iter = DataLoader(trainset, batch_size=50, collate_fn=collate_fn, sampler=train_sampler)
#test_iter = DataLoader(testset, batch_size=10, collate_fn=collate_fn, sampler=test_sampler)
#val_iter = DataLoader(valset, batch_size=10, collate_fn=collate_fn)


tgt_max_len = Pre.tgt_max_len
tgt_index2word = Pre.tgt_index2word
src_vocab = len(Pre.src_word2index)
tgt_vocab = len(Pre.tgt_word2index)
pad_index = 1


model = make_model(src_vocab, tgt_vocab).cuda()
model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[args.local_rank], output_device=args.local_rank)

crit = LabelSmoothing(tgt_vocab, pad_index, 0.1).cuda(args.local_rank)

epoch = 60
lr = 0.0001
optimizer = torch.optim.Adam(model.parameters(), lr=lr, betas=(0.9, 0.98))
scheduler = ExponentialLR(optimizer, 0.05**(1/epoch))
num_beams = 8
predict_file = 'BPEdata/pre'
model_train(model, crit, optimizer, train_iter, pad_index, tgt_vocab, epoch, scheduler)
dist.barrier()
if args.local_rank == 0:
    torch.save(model.module.state_dict(), 'model.pth')
    print('模型已保存')
