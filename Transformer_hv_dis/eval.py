import torch
def eval_beam_search(test_iters, model, tgt_max_len, num_beams,
                     predict_file, tgt_index2word, pad_index=1):
    model.cuda()
    model.eval()
    for batch in test_iters:
        src = batch[0].cuda()
        src_key_padding_mask = src.ne(pad_index).unsqueeze(-2).cuda()
        decoded = model.beam_search_decode(src, src_key_padding_mask, tgt_max_len, num_beams, bos_id = 2, eos_id = 3, pad_id = 1)
        # decoded shape batch_size*num_return x max_sent_len
        with open(predict_file, mode='a', encoding='utf-8') as f:
            for sent in decoded:
                f.write(' '.join([tgt_index2word[index.item()] if index.item() not in [2,3,1] else '' for index in sent]))
                f.write('\n')
        print('ok')