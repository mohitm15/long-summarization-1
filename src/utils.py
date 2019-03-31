import torch
import numpy as np
from config import config


def batch2input(batch, use_cuda):
    inputs = []
    inputs_oov = []
    sec_mask = []
    for article in batch.articles:
        inp = []
        inp_oov = []
        for sec in article.secs:
            inp.extend(sec.word_ids)
            inp_oov.extend(sec.word_ids_oov)
        inputs.append(inp)
        inputs_oov.append(inp_oov)
        sec_mask.append(article.sec_mask)

    enc_input = torch.LongTensor(inputs)
    enc_mask = enc_input.ne(0).float().requires_grad_()
    sec_mask = torch.LongTensor(sec_mask).float().requires_grad_()

    enc_lens = batch.enc_lens
    zeros_oov = None
    enc_input_oov = None

    batch_size = len(batch)
    if config.pointer:
        enc_input_oov = torch.LongTensor(inputs_oov)
    if batch.max_oov > 0:
        zeros_oov = torch.zeros(
            (batch_size, batch.max_oov), requires_grad=True)

    context = torch.zeros(
        (batch_size, 2 * config.hidden_dim), requires_grad=True)

    cov = None
    if config.cov:
        cov = torch.zeros(enc_input.size(), requires_grad=True)

    if use_cuda:
        enc_input = enc_input.cuda()
        enc_mask = enc_mask.cuda()
        context = context.cuda()
        sec_mask = sec_mask.cuda()

        if enc_input_oov is not None:
            enc_input_oov = enc_input_oov.cuda()
        if zeros_oov is not None:
            zeros_oov = zeros_oov.cuda()
        if cov is not None:
            cov = cov.cuda()

    return enc_input, enc_mask, sec_mask, enc_lens, context, cov, enc_input_oov, zeros_oov


def batch2output(batch, use_cuda):
    targets = []
    targets_oov = []
    for abstract in batch.abstracts:
        targets.append(abstract.word_ids)
        targets_oov.append(abstract.word_ids_oov)

    dec_input = torch.LongTensor(targets)
    dec_mask = dec_input.ne(0).float().requires_grad_()
    dec_lens = batch.dec_lens
    dec_len = max(dec_lens)
    dec_lens = torch.Tensor(dec_lens).float().requires_grad_()
    target = torch.LongTensor(targets_oov)

    if use_cuda:
        dec_input = dec_input.cuda()
        dec_mask = dec_mask.cuda()
        dec_lens = dec_lens.cuda()
        target = target.cuda()

    return dec_input, dec_mask, target, dec_len, dec_lens

def normalize(tensor):
    return tensor / tensor.sum(1).unsqueeze(1)

def update_loss(loss_batch, loss, weight=0.99):
    if loss == 0:
        return loss_batch
    else:
        return loss*weight + (1-weight)*loss_batch

def init(rnn):
    weights = []
    biases = []
    for wns in lstm._all_weights:
        for wn in wns:
            if wn.startswith('weight'):
                weights.append(wn)
            elif wn.startswith('bias'):
                biases.append(wn)

    for w in weights:
        weight = getattr(rnn, w)
        weight.data.uniform_(-0.02, 0.02)

    for b in biases:
        bias = getattr(rnn, b)
        bias.data.fill_(0.)
        bias.data[int(bias.size(0)/4):int(bias.size(0)/2)].fill_(1.)