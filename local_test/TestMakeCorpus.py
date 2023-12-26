import torch
from fairseq.data import dictionary
import torch.nn.functional as F
import numpy as np

if __name__ == '__main__':
    input_string = torch.tensor(
        [[0, 23, 45, 300, 3120, 1, 2, 0, 56, 12, 7, 100, 1, 2, 0, 75, 543, 213, 932, 214, 2, 0, 546, 765, 300, 11, 154,
          2]])
    dict = dictionary.Dictionary()
    start_tag = (input_string == dict.bos_index).int()
    tok_tags = torch.cumsum(start_tag, dim=1)
    local_attn_mask = None
    local_attn_mask = tok_tags.unsqueeze(1) == tok_tags.unsqueeze(2)
    local_attn_mask &= 0 != tok_tags.unsqueeze(2)
    unique_mask = torch.unique(local_attn_mask, dim=1)
    final_output = []
    for i, x in enumerate(unique_mask.squeeze(0)):
        element = input_string.squeeze(0)[x]
        final_output.append(element)
    corpus = (torch.stack(final_output)).float()
    x_cosine_similarity = F.cosine_similarity(corpus[None, :, :], corpus[:, None, :], dim=-1)
    # This should print the same matrix as above.
    print(x_cosine_similarity)
    mask_tag = (x_cosine_similarity > 0.82)
    length_sentence = corpus.shape[1]
    new_local_attn_mask = mask_tag.repeat_interleave(length_sentence, dim=1).repeat_interleave(length_sentence, dim=0)
    print(mask_tag)
    print(new_local_attn_mask)


def token2corpus(dict, tokens):
    start_tag = (tokens == dict.bos_index).int()
