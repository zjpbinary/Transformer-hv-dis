from lodaBPEdata import *
from eval import *
from makemodel import *

num_beams = 8
predict_file = 'BPEdata/pre'
tgt_max_len = Pre.tgt_max_len
tgt_index2word = Pre.tgt_index2word
test_iter = DataLoader(testset, batch_size=10, collate_fn=collate_fn)
src_vocab = len(Pre.src_word2index)
tgt_vocab = len(Pre.tgt_word2index)
model = make_model(src_vocab, tgt_vocab).cuda()

model.load_state_dict(torch.load('model.pth', map_location='cuda:0'))
eval_beam_search(test_iter, model, tgt_max_len, num_beams, predict_file, tgt_index2word, pad_index=1)