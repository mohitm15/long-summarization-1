from __future__ import division
import torch
import torch.nn as nn
import torch.nn.functional as F
import utils
from config import config
from numpy import random

class SectionAttention(nn.Module):
    def __init__(self):
        super(SectionAttention, self).__init__()
        self.hidden_dim = config.hidden_dim
        self.feat_dec = nn.Linear(self.hidden_dim * 2, self.hidden_dim * 2)
        self.feat_sec = nn.Linear(self.hidden_dim * 2, self.hidden_dim * 2, bias=False)
        self.score = nn.Linear(self.hidden_dim * 2, 1, bias=False)
        self.softmax = nn.Softmax(dim=1)

    def forward(self, dec_hidden, enc_sec_outputs, sec_mask):
        batch, secL, dim = enc_sec_outputs.shape

        enc_sec_feature = self.feat_sec(enc_sec_outputs.view(-1, self.hidden_dim*2))
        dec_feature = self.feat_dec(dec_hidden).unsqueeze(1).repeat(1, secL, 1).view(-1, dim)
        att_features = enc_sec_feature + dec_feature
        score = self.score(torch.tanh(att_features)).view(-1, secL)

        attn = self.softmax(score)*sec_mask
        attn = utils.normalize(attn)
        return attn

class WordAttention(nn.Module):
    def __init__(self):
        super(WordAttention, self).__init__()
        self.hidden_dim = config.hidden_dim
        self.feat = nn.Linear(self.hidden_dim * 2, self.hidden_dim * 2)
        self.score_v = nn.Linear(self.hidden_dim * 2, 1, bias=False)
        self.softmax = nn.Softmax(dim=1)
        self.use_coverage = config.cov
        if self.use_coverage:
            self.feat_cov = nn.Linear(1, config.hidden_dim * 2, bias=False)

    def forward(self, dec_hidden, enc_output, enc_feature, enc_mask, sec_attn, coverage):
        batch, src_len, dim = enc_output.shape

        dec_feature = self.feat(dec_hidden).unsqueeze(1).repeat(1, src_len, 1).view(-1, dim)
        att_features = enc_feature + dec_feature
        if config.cov:
            coverage_feature = self.feat_cov(coverage.view(-1, 1))
            att_features = att_features + coverage_feature

        secL = sec_attn.size(1)
        wordL = int(src_len/secL)
        sec_attn = sec_attn.unsqueeze(2).repeat(1,1,wordL).view(batch,-1)

        score = self.score_v(torch.tanh(att_features)).view(-1, src_len)
        attn = torch.einsum("bl,bl->bl",sec_attn, score)
        attn = self.softmax(score) * enc_mask
        attn = utils.normalize(attn)

        context = torch.bmm(attn.unsqueeze(1), enc_output).view(-1, self.hidden_dim*2)
        if self.use_coverage:
            coverage = attn + coverage

        return attn, context, coverage

class Encoder(nn.Module):
    def __init__(self):
        super(Encoder, self).__init__()
        self.emb_dim = config.emb_dim
        self.hidden_dim = config.hidden_dim
        self.dropout = nn.Dropout(config.drop_out)

        self.rnn_word = nn.LSTM(self.emb_dim, self.hidden_dim, num_layers=config.enc_layers, bidirectional=config.enc_bidi, batch_first=True)
        self.rnn_sec = nn.LSTM(self.hidden_dim, self.hidden_dim, num_layers=config.enc_layers, bidirectional=config.enc_bidi, batch_first=True)
        utils.init(self.rnn_word)
        utils.init(self.rnn_sec)

        self.sec = nn.Linear(self.hidden_dim * 2, self.hidden_dim)
        self.linear_hidden = nn.Linear(self.hidden_dim * 2, self.hidden_dim)
        self.linear_hidden.weight.data.normal_(std=1e-4)
        self.linear_hidden.bias.data.normal_(std=1e-4)

        self.linear_cell = nn.Linear(self.hidden_dim * 2, self.hidden_dim)
        self.linear_cell.weight.data.normal_(std=1e-4)
        self.linear_cell.bias.data.normal_(std=1e-4)

    def concat_tensor(self, tensor):
        return torch.cat((tensor[0], tensor[1]),1)

    def reduce_states(self, state):
        hidden, cell = state
        hidden, cell = self.dropout(hidden), self.dropout(cell)
        hidden, cell = self.concat_tensor(hidden), self.concat_tensor(cell)
        hidden = F.relu(self.linear_hidden(hidden)).unsqueeze(0)
        cell = F.relu(self.linear_cell(cell)).unsqueeze(0)
        return (hidden, cell)

    def forward(self, input, seq_lens, sec_lens, secL, wordL):
        enc_input = nn.utils.rnn.pack_padded_sequence(input, seq_lens, batch_first=True)
        enc_outputs, _ = self.rnn_word(enc_input)
        enc_outputs = nn.utils.rnn.pad_packed_sequence(enc_outputs, batch_first=True)[0].contiguous()

        sec_input = enc_outputs.view(input.size(0), secL, wordL, -1)[:,:,-1,:]
        sec_input = self.sec(sec_input)
        packed_sec = nn.utils.rnn.pack_padded_sequence(sec_input, sec_lens, batch_first=True)
        enc_sec_outputs, hidden = self.rnn_sec(packed_sec)
        enc_sec_outputs = nn.utils.rnn.pad_packed_sequence(enc_sec_outputs, batch_first=True)[0].contiguous()

        hidden = self.reduce_states(hidden)

        return enc_outputs, enc_sec_outputs, hidden

class Decoder(nn.Module):
    def __init__(self):
        super(Decoder, self).__init__()
        self.emb_dim = config.emb_dim
        self.hidden_dim = config.hidden_dim
        self.attn_word = WordAttention()
        self.attn_sec = SectionAttention()
        self.combine_context = nn.Linear(self.hidden_dim * 2 + self.emb_dim, self.emb_dim)
        self.softmax = nn.Softmax(dim=1)
        self.rnn = nn.LSTM(self.emb_dim, self.hidden_dim, num_layers=1, batch_first=True, bidirectional=False)
        utils.init(self.rnn)

        if config.pointer:
            self.sigmoid = nn.Sigmoid()
            self.linear_pointer = nn.Linear(self.emb_dim + self.hidden_dim*4, 1)


        self.feat = nn.Linear(self.hidden_dim * 2, self.hidden_dim * 2, bias=False)
        self.linear = nn.Linear(self.hidden_dim*3, self.hidden_dim)
        self.linear.weight.data.normal_(std=1e-4)
        self.linear.bias.data.normal_(std=1e-4)

        self.vocab = nn.Linear(self.hidden_dim, config.vocab_size)
        self.vocab.weight.data.normal_(std=1e-4)
        self.vocab.bias.data.normal_(std=1e-4)


    def get_gen_prob(self, dec_state, context, embedded_inputs):
        p_gen_input = torch.cat((context, dec_state, embedded_inputs), 1)
        p_gen = self.sigmoid(self.linear_pointer(p_gen_input))
        return p_gen


    def reduce_state(self, state):
        hidden, cell = state
        hidden, cell = hidden.view(-1, self.hidden_dim), cell.view(-1, self.hidden_dim)
        combined = torch.cat((hidden, cell), 1)
        return combined
    
    def forward(self, input, enc_outputs, enc_sec_output, enc_mask, sec_mask, 
                hidden, prev_context, cov, zeros_oov, enc_input_oov):
        input = self.combine_context(torch.cat((input, prev_context), 1))
        dec_out, hidden = self.rnn(input.unsqueeze(1), hidden)
        enc_feature = self.feat(enc_outputs.view(-1, self.hidden_dim*2))

        dec_state = self.reduce_state(hidden)
        sec_attn = self.attn_sec(dec_state, enc_sec_output, sec_mask)
        attn, context, cov = self.attn_word(dec_state, enc_outputs, enc_feature, enc_mask, sec_attn, cov)
        output = torch.cat((dec_out.view(-1, self.hidden_dim), context), 1)
        output = self.vocab(self.linear(output))
        output = self.softmax(output)

        if config.pointer:
            p_gen = self.get_gen_prob(dec_state, context, input)
            attn = (1-p_gen) * attn
            output = p_gen * output
            if zeros_oov is not None:
                output = torch.cat((output, zeros_oov), 1)
            output.scatter_add_(1, enc_input_oov, attn)

        return output, hidden, context, attn, cov

class Summarizer(nn.Module):
    def __init__(self, tie_emb=True):
        super(Summarizer, self).__init__()
        self.encoder = Encoder()
        self.decoder = Decoder()
        self.embedding = nn.Embedding(config.vocab_size, config.emb_dim)
        self.embedding.weight.data.normal_(std=1e-4)

    def get_cov_loss(cov, attn):
        cov_loss = torch.min(cov, attn).sum(1) * config.cov_loss_wt
        return cov_loss


    def forward(self, batch):
        enc_input, enc_mask, sec_mask, enc_lens, context, cov, enc_input_oov, zeros_oov = utils.batch2input(batch, config.cuda)
        dec_input, dec_mask, target, dec_len, dec_lens = utils.batch2output(batch, config.cuda)
        enc_input = self.embedding(enc_input)
        dec_input = self.embedding(dec_input)

        enc_outputs, enc_sec_outputs, hidden = self.encoder(enc_input, enc_lens, batch.sec_lens, batch.sec_num, batch.sec_len)

        losses, preds = [], []
        dec_steps = min(dec_len, config.max_dec_len)
        for t in range(dec_steps):
            output, hidden,  context, attn, cov_t = self.decoder(dec_input[:, t, :], enc_outputs, enc_sec_outputs, enc_mask, sec_mask, hidden, context, cov, zeros_oov, enc_input_oov)
            preds.append(output[0].argmax().item())
            target_prob = torch.gather(output, 1, target[:, t].unsqueeze(1)).squeeze() + config.eps
            loss_t = -torch.log(target_prob)
            if config.cov:
                loss_t += self.get_cov_loss(cov, attn)
                cov = cov_t
                
            loss_t *=  dec_mask[:, t]
            losses.append(loss_t)

        losses = torch.stack(losses, 1).sum(1)/dec_lens
        return losses.mean(), preds
