
import torch.nn as nn
import torch.nn.functional as F
from modules.subseqmask import *
from models.beamsearch import *
class EncoderDecoder(nn.Module):
    def __init__(self, encoder, decoder, src_embed, tgt_embed, d_model, vocab_size):
        super(EncoderDecoder, self).__init__()
        self.tgt_vocab_size = vocab_size
        self.encoder = encoder
        self.decoder = decoder
        self.src_embed = src_embed
        self.tgt_embed = tgt_embed
        self.proj = nn.Linear(d_model, vocab_size)
    def encode(self, src, src_mask):
        return self.encoder(self.src_embed(src), src_mask)
    def decode(self, memory, src_mask, tgt, tgt_mask):
        return self.decoder(self.tgt_embed(tgt), memory, src_mask, tgt_mask)
    def forward(self, src, tgt, src_mask, tgt_mask):
        return F.log_softmax(self.proj(self.decode(self.encode(src, src_mask), src_mask, tgt, tgt_mask)), dim=-1)

    def beam_search_decode(self, src, src_key_padding_mask, tgt_max_len, num_beams,
                           bos_id=2, eos_id=3, pad_id=1):
        # shape src: batch_size x src_max_len
        # shape src_key_padding_mask: batch_size x max_len
        batch_size, _, src_max_len = src_key_padding_mask.shape

        # 建立beam容器，每个样本一个
        generated_hyps = [BeamHypotheses(num_beams, tgt_max_len, 0.7) for _ in range(batch_size)]
        # 初始化beam得分
        beam_scores = torch.zeros((batch_size, num_beams)).cuda()
        # shape batch_size x num_beams
        beam_scores = beam_scores.view(-1)
        # 记录每个样本是否完成生成， 共batch_size
        done = [False for _ in range(batch_size)]
        # 为了并行计算, 一次性生成batch_size*num_beams个序列
        # 第一步填入bos_id, input_ids 记录结果
        input_ids = torch.full((batch_size * num_beams, 1), bos_id, dtype=torch.long).cuda()
        # 当前长度设为1
        cur_len = 1

        # 生成memory,为Transformer的decoder做准备
        memory = self.encode(src, src_key_padding_mask)
        # memory shape: batch_size x src_max_len x d_model
        # 注意需要把memory扩展为 (batch_size*num_beams) x src_max_len x d_model
        mem = torch.cat([memory] * num_beams, 1)
        mem = mem.contiguous().view(batch_size * num_beams, src_max_len, -1)
        src_mask = torch.cat([src_key_padding_mask]*num_beams, 1)
        src_mask = src_mask.contiguous().view(batch_size * num_beams, 1, src_max_len)
        # 循环解码
        while cur_len < tgt_max_len:
            # print(input_ids.shape)
            # 首先计算解码器的输出

            tgt_mask = subsequent_mask(cur_len).cuda()
            output = self.decode(mem, src_mask, input_ids, tgt_mask=tgt_mask)
            output = self.proj(output)

            # output shape (batch_size*num_beams) x cur_len x tgt_vocab_size
            # 取最后一个时间步的各token的概率 shape：(batch_size*num_beams) x tgt_vocab_size
            scores = next_token_logits = F.log_softmax(output[:, -1, :], dim=-1)
            # 计算序列概率，取了log所以直接加 shape: (batch_size*num_beams) x tgt_vocab_size
            next_scores = scores + beam_scores[:, None].expand_as(scores)
            # next_scores变形，方便计算 shape: batch_size x num_beams*vocab_size
            next_scores = next_scores.view(batch_size, num_beams * self.tgt_vocab_size)
            # 取出得分最高的2*k个token, 只取k个可能导致后面不足num_beams个
            next_scores, next_tokens = torch.topk(next_scores, 2 * num_beams, dim=1, largest=True, sorted=True)
            # next_scores, next_tokens, shape； batch_size x num_beams
            # 下一个时间步整个batch的beam列表
            # 列表中每个元素是（分数，token_id, beam_id）
            next_batch_beam = []
            # 对每个样本进行扩展
            for batch_idx in range(batch_size):
                # 检查样本是否已经生成结束
                if done[batch_idx]:
                    # 对于已经结束的句子，待添加的是pad_token
                    next_batch_beam.extend([(0, pad_id, 0)] * num_beams)
                    # 对下一个样本进行操作
                    continue
                # 当前样本下一个时间步的beam列表
                next_sent_beam = []
                '''
                对于未结束的样本需要找到分数最高的num_beams个扩展
                next_scores 和 next_token 是对应的
                next_scores已经被排序
                '''
                for beam_token_rank, (beam_token_id, beam_token_score) in enumerate(
                        zip(next_tokens[batch_idx], next_scores[batch_idx])):
                    beam_id = beam_token_id // self.tgt_vocab_size
                    token_id = beam_token_id % self.tgt_vocab_size
                    effective_beam_id = batch_idx * num_beams + beam_id
                    # 若出现EOS 说明已经完成生成完整句子,存入generated_hyp
                    if (eos_id is not None) and (token_id.item() == eos_id):
                        is_beam_token_worse_than_top_num_beams = beam_token_rank >= num_beams
                        if is_beam_token_worse_than_top_num_beams:
                            continue
                        # 装入
                        generated_hyps[batch_idx].add(input_ids[effective_beam_id].clone(), beam_token_score.item())
                    # 非eos
                    else:
                        next_sent_beam.append((beam_token_score, token_id, effective_beam_id))
                    # 只需要扩展到num_beams个
                    if len(next_sent_beam) == num_beams:
                        break
                # 更新句子的完成状态，有两种情况
                # 1.已经记录过样本结束
                # 2.新的结果没有使结果改善
                done[batch_idx] = done[batch_idx] or generated_hyps[batch_idx].is_done(
                    next_scores[batch_idx].max().item(), cur_len=cur_len)
                # 把当前样本的结果添加到batch结果的后面
                next_batch_beam.extend(next_sent_beam)
            # 所有样本都生成完毕，则直接退出
            if all(done):
                break
            # 把三元组还原成三个独立列表
            # shape batch_size x num_beams
            beam_scores = beam_scores.new([x[0] for x in next_batch_beam])
            beam_tokens = input_ids.new([x[1] for x in next_batch_beam])
            beam_idx = input_ids.new([x[2] for x in next_batch_beam])
            # 准备下一时刻的解码器输入
            # 取出被实际扩展的beam
            input_ids = input_ids[beam_idx, :]
            # 在这些beam后面街上新生成的token
            input_ids = torch.cat([input_ids, beam_tokens.unsqueeze(1)], dim=-1)
            # 更新当前长度
            cur_len += 1
            # 长度循环结束
        # 将未结束的生成结果结束，并置入容器中
        for batch_idx in range(batch_size):
            if done[batch_idx]:
                continue
            # 把结果加入到generated_hyps容器
            for beam_id in range(num_beams):
                effective_beam_id = batch_idx * num_beams + beam_id
                final_score = beam_scores[effective_beam_id].item()
                final_tokens = input_ids[effective_beam_id]
                generated_hyps[batch_idx].add(final_tokens, final_score)
        # 选择最终输出
        # 规定每个样本返回几个句子
        output_num_return_sequences_per_batch = 1
        output_batch_size = output_num_return_sequences_per_batch * batch_size
        # 记录每个返回句子的长度，便于pad
        sent_lengths = input_ids.new(output_batch_size)
        best = []
        # 对每个样本选出 要求返回数量 的句子
        for i, hypotheses in enumerate(generated_hyps):
            sorted_hyps = sorted(hypotheses.beams, key=lambda x: x[0])
            for j in range(output_num_return_sequences_per_batch):
                effective_batch_idx = output_num_return_sequences_per_batch * i + j
                best_hyp = sorted_hyps.pop()[1]
                sent_lengths[effective_batch_idx] = len(best_hyp)
                best.append(best_hyp)
        # 如果长短不一则pad句子，使得最后返回的结果的长度一样
        if sent_lengths.min().item() != sent_lengths.max().item():
            # 找到句子长度
            sent_max_len = min(tgt_max_len, sent_lengths.max().item() + 1)
            # 先把输出矩阵填满pad
            decoded = input_ids.new(output_batch_size, sent_max_len).fill_(pad_id)
            # 填入真正的内容
            for i, hypo in enumerate(best):
                decoded[i, :sent_lengths[i]] = hypo
                # 填上eos
                if sent_lengths[i] < tgt_max_len:
                    decoded[i, sent_lengths[i]] = eos_id
        else:
            # 所有生成序列都没结束
            decoded = torch.stack(best).type(torch.long)
        # 返回的结果中是包含bos的
        return decoded
        # shape batch_size*num_return x sent_max_len
