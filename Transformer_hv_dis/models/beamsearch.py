class BeamHypotheses(object):
    def __init__(self, num_beams, max_length, length_penalty):
        self.max_len = max_length - 1
        self.length_penalty = length_penalty
        self.num_beams = num_beams
        self.beams = []
        self.worst_score = 1e9
    # 返回列表中假设的个数
    def __len__(self):
        return len(self.beams)
    # 向列表中添加假设
    def add(self, hyp, sum_logprobs):
        # 长度惩罚
        score = sum_logprobs / len(hyp) ** self.length_penalty
        # 数量不够和得分更高是进行更新
        if len(self) < self.num_beams or score>self.worst_score:
            # 将得分与序列 加入到列表中
            self.beams.append((score, hyp))
            if len(self) > self.num_beams:
                # 数量饱和删除得分最低的
                sorted_score = sorted([(s, idx) for idx, (s,_) in enumerate(self.beams)])
                del self.beams[sorted_score[0][1]]
                self.worst_score = sorted_score[1][0]
            else:
                self.worst_score = min(score, self.worst_score)
    # 判断样本是否已经完成生成
    def is_done(self, best_sum_logprobs, cur_len=None):
        if len(self) < self.num_beams:
            return False
        else:
            if cur_len is None:
                cur_len = self.max_len
            cur_score = best_sum_logprobs / cur_len ** self.length_penalty
            # 是否最高分比当前保存的最低分还差
            ret = self.worst_score >= cur_score
            return ret