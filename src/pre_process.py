"""
March 2019
Xinru Yan

Data processing for NN
"""
from __future__ import unicode_literals
from dataclasses import dataclass
from itertools import islice
from typing import Dict, NamedTuple, List, Union, Optional, TypeVar, Iterator

from config import config
import random
import json
import numpy as np
np.random.seed(config.seed)

# https://docs.python.org/3/library/typing.html#typing.TypeVar
T = TypeVar('T')
S = TypeVar('S', int, str)


class Labeler:
    def __init__(self):
        self._w2i: Dict[str, int] = {}
        self._i2w: Dict[int, str] = {}

    def __len__(self):
        assert len(self._w2i) == len(self._i2w), 'Size mismatch! This should never happen.'
        return len(self._w2i)

    def __getitem__(self, item: Union[int, str]) -> Optional[Union[int, str]]:
        return self.get(item, default=None)

    def __contains__(self, item: Union[int, str]) -> bool:
        if isinstance(item, int):
            return item in self._i2w
        elif isinstance(item, str):
            return item in self._w2i

    def get(self, item: S, default=None) -> Optional[S]:
        """Converts a word to its' id or vice versa.

        Args:
            item: The word id or the word.

        Returns:
            The id of the word or the word for the id. Returns None if oov.
        """
        if isinstance(item, int):
            return self.get_word(item, default)
        elif isinstance(item, str):
            return self.get_id(item, default)

    def get_word(self, wid: int, default=None) -> str:
        return self._i2w.get(wid, default)

    def get_id(self, word: str, default=None, offset=0) -> int:
        if offset == 0:
            return self._w2i.get(word, default)
        else:
            wid = self._w2i.get(word)
            if wid is not None:
                return wid + offset
            else:
                return default

    def add(self, word) -> int:
        """Adds a word to the vocab.

        Args:
            word: The word to add.

        Returns:
            the id of the word added.
        """
        if word not in self._w2i:
            # Add word since it is a new one.
            self._w2i[word] = len(self._w2i)
            self._i2w[len(self._i2w)] = word
        return self[word]


class Vocab(Labeler):
    UNK = '<UNK>'
    PAD = '<PAD>'
    START_DECODING = '<START>'
    STOP_DECODING = '<STOP>'
    SOS = '<S>'
    EOS = '</S>'

    SPECIAL_WORDS = [PAD, UNK, START_DECODING, STOP_DECODING, SOS, EOS]

    def __init__(self, *vocab_files):
        super().__init__()

        for word in self.SPECIAL_WORDS:
            self.add(word)

        for file in vocab_files:
            with open(file, 'r') as fp:
                for line in fp:
                    word, _ = line.split()  # each line should be in the format "word freq"
                    assert word not in self.SPECIAL_WORDS, f'Vocab can\'t contain any of: {self.SPECIAL_WORDS}'
                    self.add(word)


@dataclass
class Sentence:
    words: List[str]
    word_ids: List[int]
    word_ids_oov: List[int]

    def __len__(self):
        assert len(self.word_ids) == len(self.words), 'Parallel arrays must be the same length!'
        return len(self.words)

    def padded(self, length: int, padding_word: str, padding_id: int) -> 'Sentence':
        assert len(self) <= length, 'Padding length must be equal to or greater than the length of the sentence.'
        needed_padding = length-len(self)
        return Sentence(words=self.words + [padding_word]*needed_padding,
                        word_ids=self.word_ids + [padding_id]*needed_padding,
                        word_ids_oov=self.word_ids_oov + [padding_id]*needed_padding)


class Section(NamedTuple):
    name: str
    words: List[str]
    word_ids: List[int]
    word_ids_oov: List[int]

    def __len__(self):
        assert len(self.word_ids) == len(self.words), 'Parallel arrays must be the same length!'
        return len(self.words)

    def padded(self, num_words: int, padding_word: str, padding_id: int) -> 'Section':
        needed_section_padding = min(num_words, config.max_sec_len) - len(self)
        needed_section_padding = max(needed_section_padding, 0)
        assert needed_section_padding >= 0, 'Padding length must be equal to or greater than the number of sentences.'

        words = self.words[:config.max_sec_len] + [padding_word] * needed_section_padding
        word_ids = self.word_ids[:config.max_sec_len] + [padding_id] * needed_section_padding
        word_ids_oov = self.word_ids_oov[:config.max_sec_len] + [padding_id] * needed_section_padding
        return Section(name=self.name, words=words, word_ids=word_ids, word_ids_oov=word_ids_oov)


class Article(NamedTuple):
    secs: List[Section]
    sec_mask: List[int]
    oovv: Labeler
    id: str

    def __len__(self):
        return len(self.secs)

    @classmethod
    def from_obj(cls, sects_obj: List[List[str]], sect_names: List[str], id:str, vocab: Vocab) -> 'Article':
        secs = []
        oovv = Labeler()
        num_sec = 0
        for sec, name in zip(sects_obj, sect_names):
            if num_sec >= config.max_num_sec:
                break
            sec_words = []
            sec_word_ids = []
            sec_word_ids_oovs = []
            for sent in sec:
                words = sent.split()
                words_ids = []
                words_ids_oov = []
                for word in words:
                    wid = wid_oov = vocab[word]
                    if wid is None:
                        wid_oov = oovv.add(word) + len(vocab)
                        wid = vocab["<UNK>"]
                    words_ids.append(wid)
                    words_ids_oov.append(wid_oov)
                sec_words.extend(words)
                sec_word_ids.extend(words_ids)
                sec_word_ids_oovs.extend(words_ids_oov)
            secs.append(Section(name=name, words=sec_words, word_ids=sec_word_ids, word_ids_oov=sec_word_ids_oovs))
            num_sec += 1
        return Article(secs=secs, oovv=oovv, sec_mask=[], id=id)

    @property
    def longest_word_len(self) -> int:
        return max(len(sec) for sec in self.secs)

    def padded(self, word_length: int, sec_length: int, vocab: Vocab) -> 'Article':
        """First pad the sentences in each article, then pad the sections

        Args:
            num_sents: max number of sents
            sent_length: max sent length
            sec_length: max sec length
            vocab: vocab

        Returns:
            Padded Article
            The article has 'max sec length' of sections
            Each sentence is 'max sent length' long
        """
        padding_word = vocab.PAD
        padding_id = vocab[padding_word]
        secs = []
        for sec in self.secs:
            if len(sec) == 0:
                continue
            secs.append(sec.padded(word_length, padding_word=padding_word, padding_id=padding_id))
        # secs = [sec.padded(word_length, padding_word=padding_word, padding_id=padding_id) for sec in self.secs]
        needed_section_padding = sec_length - len(secs)
        sec_mask = [1] * (sec_length-needed_section_padding) + [0] * needed_section_padding
        assert needed_section_padding >= 0, 'Padding length must be equal to or greater than the number of sections.'
        for _ in range(needed_section_padding):
            name = 'PAD_SEC'
            sec = Section(name=name, words=[], word_ids=[], word_ids_oov=[])
            secs.append(sec.padded(word_length, padding_word=padding_word, padding_id=padding_id))
        return Article(secs=secs, oovv=self.oovv, sec_mask=sec_mask, id=self.id)


class Abstract(NamedTuple):
    words: List[str]
    word_ids: List[int]
    word_ids_oov: List[int]

    def __len__(self):
        assert len(self.words) == len(self.word_ids), "need to match"
        return len(self.words)


    @classmethod
    def from_obj(cls, obj: List[str], article: Article, vocab: Vocab) -> 'Abstract':
        abstract_words = ["<START>"]
        abstract_word_ids = [vocab[vocab.START_DECODING]]
        abstract_word_ids_oov = [vocab[vocab.START_DECODING]]

        for sent in obj:
            words = sent.split()
            words_ids: List[int] = []
            words_ids_oov: List[int] = []
            for word in words:
                # find id from vocab, if not there get id from article's oov vocab, if not there use the UNK word id.
                word_id = vocab.get_id(word, vocab.get_id(vocab.UNK))
                word_id_oov = vocab.get_id(word, article.oovv.get_id(word, vocab.get_id(vocab.UNK), offset=len(vocab)))
                words_ids.append(word_id)
                words_ids_oov.append(word_id_oov)
            abstract_words.extend(words)
            abstract_word_ids.extend(words_ids)
            abstract_word_ids_oov.extend(words_ids_oov)
        abstract_words.append("<STOP>")
        abstract_word_ids.append(vocab[vocab.STOP_DECODING])
        abstract_word_ids_oov.append(vocab[vocab.STOP_DECODING])
        return Abstract(words=abstract_words, word_ids=abstract_word_ids, word_ids_oov=abstract_word_ids_oov)

    def padded(self, num_words: int, vocab: Vocab) -> 'Abstract':
        needed_words_padding = min(num_words, config.max_dec_len) - len(self)
        needed_words_padding = max(needed_words_padding, 0)
        assert needed_words_padding >= 0, 'Padding length must be equal to or greater than the number of sentences.'
        padding_word = vocab.PAD
        padding_id = vocab[padding_word]
        words = self.words[:config.max_dec_len] + [padding_word] * needed_words_padding
        word_ids = self.word_ids[:config.max_dec_len] + [padding_id] * needed_words_padding
        word_ids_oov = self.word_ids_oov[:config.max_dec_len] + [padding_id] * needed_words_padding
        return Abstract(words=words, word_ids=word_ids, word_ids_oov=word_ids_oov)


class Example(NamedTuple):
    article: Article
    abstract: Abstract

    def padded(self, article_sec_length: int, article_max_word_len: int, abstract_max_word_len: int, vocab: Vocab) -> 'Example':
        return Example(article=self.article.padded(article_max_word_len, article_sec_length, vocab),
                       abstract=self.abstract.padded(abstract_max_word_len, vocab))


class Batch:
    # examples: List[Example]
    articles: List[Article]
    abstracts: List[Abstract]
    article_pad_len: int
    enc_lens: List[int]
    dec_lens: List[int]
    max_oov: int
    sec_num: int
    sec_len: int
    dec_len: int

    def __init__(self, unpadded_examples: List[Example], vocab: Vocab):
        article_sec_length = max(len(example.article) for example in unpadded_examples)
        article_sec_length = min(article_sec_length, config.max_num_sec)
        self.sec_lens = [min(len(example.article),article_sec_length) for example in unpadded_examples]

        article_max_word_len = max(len(sec) for example in unpadded_examples for sec in example.article.secs)
        article_max_word_len = min(article_max_word_len, config.max_sec_len)

        abstract_max_word_len = max(len(example.abstract) for example in unpadded_examples)
        abstract_max_word_len = min(abstract_max_word_len, config.max_dec_len)

        self.article_pad_len = article_sec_length
        self.enc_lens = [min(article_max_word_len, config.max_sec_len)* min(len(example.article), config.max_num_sec) for example in unpadded_examples]
        examples = [example.padded(article_sec_length=article_sec_length,
                                        article_max_word_len=article_max_word_len,
                                        abstract_max_word_len= abstract_max_word_len,
                                        vocab=vocab) for example in unpadded_examples]
        examples, self.enc_lens, self.sec_lens = zip(*[(e,l,s) for e, l, s in sorted(zip(examples,self.enc_lens, self.sec_lens), key=lambda pair: (-pair[1], -pair[2]))])
        self.dec_lens = [min(len(example.abstract), config.max_dec_len) for example in unpadded_examples]
        self.articles = [example.article for example in examples]
        self.abstracts = [example.abstract for example in examples]
        self.max_oov = max([len(example.article.oovv) for example in examples])
        self.sec_num = min(article_sec_length, config.max_num_sec)
        self.sec_len = min(article_max_word_len, config.max_sec_len)
        self.dec_len = min(abstract_max_word_len, config.max_dec_len)

    def __len__(self):
        return len(self.articles)

    def __iter__(self):
        return iter(self.articles)

    def __repr__(self):
        return f'<Batch examples:{repr(self.articles)}>'


class DataLoader:
    def __init__(self, cfg):
        self.config = cfg

        self.vocab_file_path = config.vocab_path
        self.train_file_path = config.train_data_path
        self.test_file_path = config.decode_data_path
        print('building vocabs from the vocab file')
        self.vocab = Vocab(self.vocab_file_path)

        print(f'vocab size is {len(self.vocab)}')

    def load_data(self, *file_path) -> Iterator[Example]:
        """Loads a list of examples from the provided files.

        Args:
            *file_path: File paths to the files which to load the examples from.

        Returns:
            List of examples load from the files.
        """
        return self.__process_data(self.__load_data(*file_path))

    def __load_data(self, *file_path) -> Iterator[Dict[List[str], List[str]]]:
        for file in file_path:
            with open(file, 'r') as f:
                for idx, line in enumerate(f):
                    yield json.loads(line)

    def __process_data(self, data: Iterator[Dict[List[str], List[str]]]) -> Iterator[Example]:
        for item in data:
            article = Article.from_obj(item['sections'], item['section_names'], item['article_id'], self.vocab)
            abstract = Abstract.from_obj(item['abstract_text'], article, self.vocab)
            example = Example(article=article, abstract=abstract)
            if not (len(example.article) == 1 and example.article.longest_word_len == 0):
                yield example

    def get_training_examples(self) -> Iterator[Example]:
        return self.load_data(self.train_file_path)

    def get_test_examples(self) -> Iterator[Example]:
        return self.load_data(self.test_file_path)

def batchify(examples: Iterator[Example], batch_size: int, vocab: Vocab, repeat: bool = False) -> Iterator:
    """ batchifies the examples

    Args:
        examples: iterator of examples
        batch_size: # of examples in each batch
        vocab: the vocab to pass to each batch
        repeat: if repeat, the batch contains one unique example repeated batch_size times (here batch_size=config.beam_size);
                 otherwise the batch contains batch_size number of examples
    """
    if repeat:
        for e in examples:
            yield Batch(unpadded_examples=sorted([e for _ in range(batch_size)], key=lambda x: -min(x.article.longest_word_len, config.max_sec_len)*len(x.article)), vocab=vocab)
    else:
        ex_subset = []

        # Mini-shuffle within batch shuffle window then yield
        for e in examples:
            if len(ex_subset) >= batch_size * config.batch_shuffle_window:
                random.shuffle(ex_subset)
                for i in range(config.batch_shuffle_window):
                    yield Batch(unpadded_examples=sorted(ex_subset[batch_size*i : batch_size*(i+1)], key=lambda x: -len(x.article)), vocab=vocab)
                    # yield Batch(unpadded_examples=ex_subset[batch_size*i : batch_size*(i+1)], vocab=vocab)
                ex_subset = []
            ex_subset.append(e)

        # Flush (yield) remaining examples
        if len(ex_subset) > 0:
            random.shuffle(ex_subset)
            while True:
                ex_subset = (e for e in ex_subset) # Convert to generator
                batch_examples = list(islice(ex_subset, batch_size))
                if not batch_examples:
                    break
                yield Batch(unpadded_examples=sorted(batch_examples, key=lambda x: -min(x.article.longest_word_len, config.max_sec_len)*len(x.article)), vocab=vocab)
                # yield Batch(unpadded_examples=batch_examples, vocab=vocab)
