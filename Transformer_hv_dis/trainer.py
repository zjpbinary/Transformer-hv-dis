from modules.subseqmask import *
def model_train(model, crit, opt, train_iter, pad_index, tgt_vocab, epoch, sche):
    model.train()
    for ep in range(epoch):
        print(f'第{ep}个epoch:')
        for i, batch in enumerate(train_iter):
            opt.zero_grad()
            src = batch[0].cuda()
            tgt = batch[1].cuda()
            _, src_size = src.shape
            _, tgt_size = tgt.shape
            tgt_mask = tgt.ne(pad_index).unsqueeze(-2) & subsequent_mask(tgt_size).cuda()
            # shape batch_size x tgt_max_len x tgt_max_len
            src_mask = src.ne(pad_index).unsqueeze(-2).cuda()
            out = model.forward(src, tgt[:, :-1], src_mask, tgt_mask[:, :-1, :-1])
            # shape batch_size x tgt_max_len-1 x tgt_vocab
            loss = crit(out.contiguous().view(-1, tgt_vocab), tgt[:, 1:].contiguous().view(-1))
            loss.backward()
            opt.step()
            print(loss)
        sche.step()
