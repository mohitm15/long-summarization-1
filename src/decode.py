from pre_process import DataLoader, batchify
from config import config
from utils import batch2input
from main import get_model
import os
import torch
import pyrouge

class Hypothesis():
    def __init__(self, tokens, log_probs, hidden, betas):
        self.tokens = tokens
        self.log_probs = log_probs
        self.hidden = hidden
        self.betas = betas
    
    def extend(self, token, log_prob, hidden, beta):
        return Hypothesis(
            tokens = self.tokens + [token],
            log_probs = self.log_probs + [log_prob],
            hidden = hidden,
            betas = self.betas + [beta]
        )


class BeamSearchDecoder():
    def __init__(self, model, data):
        self.model = model
        self.data = data
        self.batches = batchify(data.get_test_examples(), config.batch_size, data.vocab, True)

    def decode(self):
        for batch in self.batches:
            # Best hypothesis
            best_hyp = beam_search(self.model, self.data.vocab, batch)

            # Convert best hypothesis back to words, removing START/STOP_DECODING tokens
            res_words = ids2words(best_hyp.tokens, self.data.vocab, batch.articles[0].oovv)
            res_words = res_words[1:]
            if res_words[-1] == self.data.vocab.STOP_DECODING:
                res_words = res_words[:-1]

            # Clean <START>, <STOP>, <S>, </S> from ref abstracts
            ref_words = batch.abstracts[0].words
            ref_words.remove("<START>")
            while "<STOP>" in ref_words:
                ref_words.remove("<STOP>")
            while "<S>" in ref_words:
                ref_words.remove("<S>")
            while "</S>" in ref_words:
                ref_words.remove("</S>")

            # Write ref & decoded summary to file, for ROUGE eval
            save_rouge(ref_words, res_words, batch.articles[0].id)
            # print('Decoded article %s' % (batch.articles[0].id,))

        # ROUGE evaluation
        r = pyrouge.Rouge155('./ROUGE-1.5.5')
        r.model_filename_pattern = '#ID#_ref.txt'
        r.system_filename_pattern = '(\w+)_dec.txt'
        r.model_dir = config.log_save_dir + '/rouge_ref'
        r.system_dir = config.log_save_dir + '/rouge_dec'

        rouge_results = r.convert_and_evaluate()
        print(rouge_results)

        return rouge_results

def beam_search(model, vocab, batch):
    with torch.no_grad():
        # Run the encoder to get the encoder hidden hiddens and decoder initial hidden
        enc_input, enc_mask, sec_mask, enc_lens, context, cov, enc_input_oov, zeros_oov = batch2input(batch, config.cuda)
        enc_input = model.embedding(enc_input)
        enc_outputs, enc_sec_outputs, hidden = model.encoder(enc_input, enc_lens, batch.sec_lens, batch.sec_num, batch.sec_len)

        # Initialize beam_size-many Hypotheses
        hyps = [Hypothesis(
            tokens = [vocab[vocab.START_DECODING]],
            log_probs = [0.0],
            hidden = (hidden[0][:,0,:], hidden[1][:,0,:]),
            betas = []
        ) for _ in range(config.beam_size)]

        results = [] # To contain complete hypotheses (i.e. those that have emitted STOP_DECODING)

        steps = 0 # Decoding step
        while steps < config.max_dec_len and len(results) < config.beam_size:

            # Run model decoder for one-step decoding
            with torch.no_grad():

                # Prepare input tensor from latest tokens
                latest_tokens = [h.tokens[-1] for h in hyps]
                latest_tokens = [
                    t if t in range(len(vocab)) else vocab[vocab.UNK]
                for t in latest_tokens] # Convert any in-article OOVs to UNKs, so that we can look up embeddings for all token

                dec_input = torch.LongTensor(latest_tokens) # (beam_size, 1)
                if config.cuda:
                    dec_input = dec_input.cuda()

                # Turn list of lstm hidden tuples into a single tuple for the whole batch; i.e. (beam_size, 1, hidden_dim)
                hidden = [h.hidden for h in hyps]
                hidden = (
                    torch.stack([h for (h, _) in hidden], dim=1),
                    torch.stack([c for (_, c) in hidden], dim=1)
                )

                # Run decoder
                dec_input = model.embedding(dec_input)
                output, hidden,  context, _, _ = model.decoder(dec_input, hidden, enc_outputs, enc_sec_outputs, enc_mask, sec_mask, context,
                                                            zeros_oov, enc_input_oov, cov)

                # Get top-k probabilities and indices
                topk_probs, topk_ids = output.topk(2*config.beam_size, dim=1)
                topk_probs = topk_probs.log()

            # Extend each Hypothesis
            all_hyps = []
            num_uniq_hyps = 1 if steps == 0 else len(hyps) # On the first step we have only one unique Hypothesis

            for i in range(num_uniq_hyps):
            # For each unique Hypothesis

                for j in range(2 * config.beam_size):
                # For each of the top 2*k Hypothesis extension options
                    new_hyp = hyps[i].extend(
                        token = topk_ids[i, j].item(),
                        log_prob = topk_probs[i, j].item(),
                        hidden = (hidden[0][:, i, :], hidden[1][:, i, :]), # (1, 1, hidden_dim),
                        beta = []
                    )

                    all_hyps.append(new_hyp)

            # Filter and collect any finished Hypotheses (w/ STOP_DECODING produced)
            hyps = [] # Hypotheses to explore for the next step
            for h in sort_hyps(all_hyps): # In order of likelihood
                if h.tokens[-1] == vocab[vocab.STOP_DECODING]:
                    # Append to final result, if not too short
                    if steps >= config.min_dec_len:
                        results.append(h)
                else:
                    # Bring to next step
                    hyps.append(h)

                if len(hyps) == config.beam_size or len(results) == config.beam_size:
                    # Either we have enough Hypotheses for the next step, or results is full
                    break

            steps += 1

        # Now we have either beam_size results or we have reached max_dec_steps
        # If we don't have any complete Hypotheses, just return all current Hypotheses as result
        if len(results) == 0:
            results = hyps

        # Return the most likely Hypothesis
        results_sorted = sort_hyps(results)
        return results_sorted[0]

def sort_hyps(hyps):
	"""
	Arg:
		hyps: List of Hypothesis objects

	Returns:
		hyps sorted by descending likelihood; namely, average log probability
	"""
	return sorted(hyps, key=lambda h: sum(h.log_probs) / len(h.tokens), reverse=True)

def ids2words(id_list, vocab, article_oovs):
	"""
	Arg: 
		id_list: List of integer ids
		vocab: Vocab object
		article_oovs: Labeler object

	Returns:
		List of words, converted from seq of ids to seq of strings
	"""
	words = []

	for i in id_list:
		if vocab[i] is not None:
			words.append(vocab[i])
		elif article_oovs[i - len(vocab)] is not None:
			words.append(article_oovs[i - len(vocab)])
		else:
			raise ValueError("Error: model produced id that doesn't correspond to any words in vocab or article-oov")

	return words

def save_rouge(ref, res_words, id):
	# Divide decoded output into sentences
	res = []
	while len(res_words) > 0:
		try:
			period_idx = res_words.index(".")
		except ValueError:
			# Text remaining that doesn't end in period
			period_idx = len(res_words)
		sent = res_words[:period_idx+1]
		res_words = res_words[period_idx+1:]
		res.append(" ".join(sent))

	res_ref = []
	while len(ref) > 0:
		try:
			period_idx = ref.index(".")
		except ValueError:
			# Text remaining that doesn't end in period
			period_idx = len(ref)
		sent = ref[:period_idx+1]
		ref = ref[period_idx+1:]
		res_ref.append(" ".join(sent))

	# Write to file
	if not os.path.isdir(config.log_save_dir + '/rouge_ref'):
		os.mkdir(config.log_save_dir + '/rouge_ref')
	if not os.path.isdir(config.log_save_dir + '/rouge_dec'):
		os.mkdir(config.log_save_dir + '/rouge_dec')
	ref_file = os.path.join(config.log_save_dir, "rouge_ref", "%s_ref.txt" % id)
	dec_file = os.path.join(config.log_save_dir, "rouge_dec", "%s_dec.txt" % id)

	with open(ref_file, "wb") as f:
		for idx, sent in enumerate(res_ref):
			f.write(sent.encode('utf-8')) if idx==len(res_ref)-1 else f.write((sent+"\n").encode('utf-8'))
	with open(dec_file, "wb") as f:
		for idx, sent in enumerate(res):
			f.write(sent.encode('utf-8')) if idx==len(res)-1 else f.write((sent+"\n").encode('utf-8'))


if __name__ == '__main__':
    model, _, _, _ = get_model(config.train_from, eval=True)
    data = DataLoader(config)
    decoder = BeamSearchDecoder(model, data)
    decoder.decode()