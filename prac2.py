import numpy as np
import torch

seq_len = 5
base_idx = np.arange(1, seq_len + 1)
print(base_idx)
emo_pos = np.concatenate([base_idx.reshape(-1, 1)] * seq_len, axis=1).reshape(1, -1)[0]
print(emo_pos)
cau_pos = np.concatenate([base_idx] * seq_len, axis=0)
print(cau_pos)

emo_pos = torch.LongTensor(emo_pos)
cau_pos = torch.LongTensor(cau_pos)


if seq_len > 4:
    emo_pos_mask = np.array(list(map(lambda x: 1 <= x <= 4, emo_pos.tolist())), dtype=int)
    cau_pos_mask = np.array(list(map(lambda x: 1 <= x <= 4, cau_pos.tolist())), dtype=int)
    cau_pos_mask = torch.BoolTensor(cau_pos_mask)
    emo_pos_mask = torch.BoolTensor(emo_pos_mask)
    emo_pos = emo_pos.masked_select(emo_pos_mask)
    cau_pos = cau_pos.masked_select(cau_pos_mask)
print(emo_pos)
print(cau_pos)



