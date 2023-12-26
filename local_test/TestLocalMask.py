import torch
from fairseq.data import dictionary

if __name__ == '__main__':
    input_string = torch.tensor(
        [[0, 100, 200, 300, 3120, 1, 2, 0, 100, 200, 300, 1, 1, 2, 0, 100, 200, 300, 1, 1, 2, 0, 100, 200, 300, 1, 1,
          2]])
    dict = dictionary.Dictionary()
    mask_tag = ~((input_string == dict.pad_index) | (input_string == dict.index('4324234')))
    start_tag = (input_string == dict.bos_index).int()
    tok_tags = torch.cumsum(start_tag, dim=1) * mask_tag.int()
    print(tok_tags.numpy())
    local_attn_mask = None
    local_attn_mask = tok_tags.unsqueeze(1) != tok_tags.unsqueeze(2)
    local_attn_mask &= 0 != tok_tags.unsqueeze(2)
    print(local_attn_mask.numpy())


def token2corpus(dict, tokens):
    start_tag = (tokens == dict.bos_index).int()
