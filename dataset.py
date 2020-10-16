import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset
import unicodedata
import re


def preprocess(text):
    nfkd_form = unicodedata.normalize("NFKD", text)
    str_ = (
        u"".join([c for c in nfkd_form if not unicodedata.combining(c)]))
    str_ = re.sub(' +', ' ', str_)
    str_ = re.sub('\n', ' ', str_)
    return str_


class BucketDataset(Dataset):

    def __init__(self, csr_questions, csr_answers, target, tokenizer, max_len=512, emb_dropout=None, crop=False):
        self.csr_questions = csr_questions
        self.csr_answers = csr_answers
        self.tokenizer = tokenizer
        self.target = target
        self.emb_dropout = emb_dropout
        self.max_len = max_len
        self.crop = crop

    def __len__(self):
        return len(self.csr_questions)

    def crop_sentence(self, x, max_len):
        if len(x) <= max_len:
            return x
        overhead = len(x) - max_len + 1
        idx = np.random.randint(0, overhead)
        x = x[idx:idx+max_len]
        return x

    def crop_middle(self, x, max_len):
        if len(x) <= max_len:
            return x
        beg_idx = max_len // 4
        end_idx = max_len - beg_idx
        x = x[:beg_idx] + x[-end_idx:]
        return x

    def __getitem__(self, idx):
        question = self.csr_questions[idx]
        answer = self.csr_answers[idx]
        if self.emb_dropout:
            title = self.tokenizer.encode(preprocess(
                question[0]), dropout=self.emb_dropout, add_special_tokens=True)
            question = self.tokenizer.encode(preprocess(
                question[1]), dropout=self.emb_dropout, add_special_tokens=False)
            answer = self.tokenizer.encode(preprocess(
                answer), dropout=self.emb_dropout, add_special_tokens=False)
        else:
            title = self.tokenizer.encode(preprocess(
                question[0]), add_special_tokens=True)
            question = self.tokenizer.encode(preprocess(
                question[1]), add_special_tokens=False)
            answer = self.tokenizer.encode(preprocess(
                answer), add_special_tokens=False)
        if self.crop:
            question = self.crop_sentence(question, self.max_len - 2 - len(title))
            answer = self.crop_sentence(answer, self.max_len - 2)
        else:
            # question = self.crop_middle(question,self.max_len - 2 - len(title))
            # answer = self.crop_middle(answer,self.max_len - 2)
            question = question[:self.max_len - 2 - len(title)]
            answer = answer[:self.max_len - 2]
        question = title + [self.tokenizer.sep_token_id] + \
            question + [self.tokenizer.sep_token_id]
        answer = [self.tokenizer.cls_token_id] + \
            answer + [self.tokenizer.sep_token_id]
        question = np.array(question)
        answer = np.array(answer)
        return question.astype(np.int64), answer.astype(np.int64), self.target[idx]


def pad_collate(inputs, pad_token_id, common_tok=1):
    inputs = list(zip(*inputs))
    qlen = int(np.ceil(max([x.shape[0]+common_tok for x in inputs[0]])/8) * 8)-common_tok
    alen = int(np.ceil(max([x.shape[0]+common_tok for x in inputs[1]])/8) * 8)-common_tok
    max_len = max(qlen, alen)
    questions = pad_sequence([torch.from_numpy(x)
                              for x in inputs[0]], batch_first=True, padding_value=pad_token_id,
                             max_len=max_len)
    answers = pad_sequence([torch.from_numpy(x)
                            for x in inputs[1]], batch_first=True, padding_value=pad_token_id,
                           max_len=max_len)
    targets = torch.from_numpy(
        np.stack(inputs[2], 0).astype(np.float32))
    return questions, answers, targets


def pad_sequence(sequences, batch_first=True, padding_value=0., max_len=None):
    # assuming trailing dimensions and type of all the Tensors
    # in sequences are same and fetching those from sequences[0]
    max_size = sequences[0].size()
    trailing_dims = max_size[1:]
    max_len = max_len or int(
        np.ceil(max([s.size(0) for s in sequences])/8) * 8)
    if batch_first:
        out_dims = (len(sequences), max_len) + trailing_dims
    else:
        out_dims = (max_len, len(sequences)) + trailing_dims

    out_tensor = sequences[0].data.new(*out_dims).fill_(padding_value)
    for i, tensor in enumerate(sequences):
        length = tensor.size(0)
        # use index notation to prevent duplicate references to the tensor
        if batch_first:
            out_tensor[i, :length, ...] = tensor
        else:
            out_tensor[:length, i, ...] = tensor

    return out_tensor
