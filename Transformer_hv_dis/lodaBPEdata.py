import torch
from torch.utils.data import Dataset, DataLoader
class Preprocess():
    def __init__(self):
        self.src_word2index = dict()
        self.tgt_word2index = dict()
        self.src_index2word = ['<unk>', '<pad>', '<bos>', '<eos>']
        self.tgt_index2word = ['<unk>', '<pad>', '<bos>', '<eos>']
        self.src_max_len = 0
        self.tgt_max_len = 0
    # 读取数据集
    def read_file(self, filename):
        with open(filename, mode='r', encoding='utf-8') as f:
            sents = [['<bos>']+sent.strip().split()+['<eos>'] for sent in f.readlines()]
        return sents
    # 读取词表
    def read_vocab(self, src_vocab_file, tgt_vocab_file):
        with open(src_vocab_file, mode='r', encoding='utf-8') as f1:
            src_vocab = [line.split()[0] for line in f1.readlines()]
        self.src_index2word.extend(src_vocab)
        for i, word in enumerate(self.src_index2word):
            self.src_word2index[word] = i
        with open(tgt_vocab_file, mode='r', encoding='utf-8') as f2:
            tgt_vocab = [line.split()[0] for line in f2.readlines()]
        self.tgt_index2word.extend(tgt_vocab)
        for i, word in enumerate(self.tgt_index2word):
            self.tgt_word2index[word] = i
    # 获取最大长度
    def get_max_len(self, sents):
        max_len = max([len(sent) for sent in sents])
        return max_len
    # 索引化
    def to_index(self, sents, lg):
        if lg == 'src':
            return [self.add_pad([self.src_word2index[token] if token in self.src_word2index else 0 for token in sent],
                                 self.src_max_len) for sent in sents]
        else:
            return [self.add_pad([self.tgt_word2index[token] if token in self.tgt_word2index else 0 for token in sent],
                                 self.tgt_max_len) for sent in sents]
    # 填充pad
    def add_pad(self, sent, max_len):
        if len(sent) >= max_len:
            return sent[:max_len]
        else:
            return sent + [1] * (max_len - len(sent))
    # 预处理
    def forward(self, src_train, tgt_train,
                src_test, tgt_test,
                src_val, tgt_val,
                src_vocab_file, tgt_vocab_file):
        src_train_sents = self.read_file(src_train)
        tgt_train_sents = self.read_file(tgt_train)
        src_test_sents = self.read_file(src_test)
        tgt_test_sents = self.read_file(tgt_test)
        src_val_sents = self.read_file(src_val)
        tgt_val_sents = self.read_file(tgt_val)

        self.read_vocab(src_vocab_file, tgt_vocab_file)

        self.src_max_len = self.get_max_len(src_train_sents)
        self.tgt_max_len = self.get_max_len(tgt_train_sents)

        src_train_sents, src_test_sents, src_val_sents = self.to_index(src_train_sents, 'src'), \
                                                         self.to_index(src_test_sents, 'src'), \
                                                         self.to_index(src_val_sents, 'src')
        tgt_train_sents, tgt_test_sents, tgt_val_sents = self.to_index(tgt_train_sents, 'tgt'), \
                                                         self.to_index(tgt_test_sents, 'tgt'), \
                                                         self.to_index(tgt_val_sents, 'tgt')
        return src_train_sents, src_test_sents, src_val_sents,\
               tgt_train_sents, tgt_test_sents, tgt_val_sents

class MyDataset(Dataset):
    def __init__(self, src, tgt):
        self.src = src
        self.tgt = tgt
    def __getitem__(self, item):
        return self.src[item], self.tgt[item]
    def __len__(self):
        return len(self.src)


src_train = 'BPEdata/train.en.40000'
tgt_train = 'BPEdata/train.de.40000'
src_test = 'BPEdata/test.en.40000'
tgt_test = 'BPEdata/test.de.40000'
src_val = 'BPEdata/val.en.40000'
tgt_val = 'BPEdata/val.de.40000'
src_vocab_file = 'BPEdata/vocab.en.40000'
tgt_vocab_file = 'BPEdata/vocab.de.40000'

Pre = Preprocess()
src_train_sents, src_test_sents, src_val_sents, \
tgt_train_sents, tgt_test_sents, tgt_val_sents = Pre.forward(src_train, tgt_train,
                                                             src_test, tgt_test,
                                                             src_val, tgt_val,
                                                             src_vocab_file, tgt_vocab_file)
def collate_fn(batch_data):
    src, tgt = list(zip(*batch_data))
    return torch.LongTensor(src), torch.LongTensor(tgt)

trainset = MyDataset(src_train_sents, tgt_train_sents)
testset = MyDataset(src_test_sents, tgt_test_sents)
valset = MyDataset(src_val_sents, tgt_val_sents)


'''
for i, batch in enumerate(train_iter):
    print(i)
    print(batch[0].shape)
'''





