"""
March 2019
Xinru Yan
This is a configuration file, change any parameters here
"""
import os
import argparse
import datetime

parser = argparse.ArgumentParser(description='Model parameters.')

parser.add_argument('-gpus', default=[], nargs='+', type=int, help="Use CUDA with listed GPUs.")
parser.add_argument('-cuda', action='store_true', default=False)
parser.add_argument('-mode', type=str, choices=['train', 'eval', 'decode'], action='store', default='train')

parser.add_argument('-hier', action='store_true', default=True)
parser.add_argument('-pointer', action='store_true', default=True)
parser.add_argument('-cov', action='store_true', default=False)
parser.add_argument('-test', action='store_true', default=False)
parser.add_argument('-verbose', action='store_true', default=True)

parser.add_argument('-log_root', type=str, action='store', default='log')
parser.add_argument('-print_interval', type=int, action='store', default=50)
parser.add_argument('-save_interval', type=int, action='store', default=1000)


parser.add_argument('-save_dir', type=str, action='store', default='test_decode', help="path to save the trained model")
parser.add_argument('-train_from', type=str, action='store', default=None, help="path to load the trained model")

parser.add_argument('-data', type=str, action='store', default='pubmed', help="choose among [arxiv, pubmed]")
parser.add_argument('-vocab_path', type=str, action='store', default='data/pubmed/vocab')
parser.add_argument('-seed', type=int, action='store', default=111)

parser.add_argument('-ep', type=int, action='store', default=10)
# parser.add_argument('-start_ep', type=int, action='store', default=1)
# parser.add_argument('-pre_ep', type=int, action='store', default=8)
parser.add_argument('-max_iterations', type=int, action='store', default=500000)

parser.add_argument('-emb_dim', type=int, action='store', default=128)
parser.add_argument('-hidden_dim', type=int, action='store', default=256)
parser.add_argument('-batch_size', type=int, action='store', default=3)
parser.add_argument('-beam_size', type=int, action='store', default=4)
parser.add_argument('-vocab_size', type=int, action='store', default=50006)
parser.add_argument('-batch_shuffle_window', type=int, action='store', default=1)

parser.add_argument('-optim', type=str, choices=['adagrad', 'adam', 'sgd'], action='store',default='adagrad', help='[adagrad|adam|sgd]')
parser.add_argument('-max_grad_norm', type=float, default=2.0, help="""gradient clipping: gradient vector norm exceeds, renormalize it norm=max_grad_norm""")
parser.add_argument('-lr', type=float, action='store', default=0.15)
parser.add_argument('-lr_coverage', type=float, action='store', default=0.15)
parser.add_argument('-adagrad_init_acc', type=float, action='store', default=0.1)
parser.add_argument('-eps', type=float, action='store', default=1e-12)

parser.add_argument('-drop_out', type=float, action='store', default=0.25)
parser.add_argument('-enc_layers', type=int, action='store', default=1)
parser.add_argument('-enc_bidi', action='store_true', default=True)
parser.add_argument('-cov_loss_wt', type=float, action='store', default=1.0, help='Weight of coverage loss (lambda in the paper).')

# limit of lengths
parser.add_argument('-max_num_sec', type=int, action='store', default=4, help='Maximum section length')

parser.add_argument('-min_dec_len', type=int, action='store', default=35, help="only applies for decode mode")
parser.add_argument('-max_dec_len', type=int, action='store', default=210)
parser.add_argument('-max_enc_len', type=int, action='store', default=2500, help='max encoder timesteps (max source token length)')
parser.add_argument('-max_sec_len', type=int, action='store', default=500, help='Truncate sections')
# parser.add_argument('-min_section_len', type=int, action='store', default=50, help='Discards short sections (commented out because it was not used in the original code)')
parser.add_argument('-max_article_len', type=int, action='store', default=2000, help='Maximum input article length')

parser.add_argument('-max_intro_len', type=int, action='store', default=400, help='Maximum introduction section length')
parser.add_argument('-max_conclusion_len', type=int, action='store', default=400, help='Maximum conclusion section length')
parser.add_argument('-max_intro_sents', type=int, action='store', default=20, help='Maximum introduction section length')
parser.add_argument('-max_conclusion_sents', type=int, action='store', default=20, help='Maximum conclusion section length')
parser.add_argument('-max_section_sents', type=int, action='store', default=20, help='Maximum section length in sentences')

config = parser.parse_args()

# in the decode mode, batch size should be the same as the beam size
if config.mode == "decode":
    config.batch_size = config.beam_size

# indexing save_dirs for model and log
now = datetime.datetime.now()
idx = f'{config.save_dir}_{now.month}{now.day}_{now.hour}{now.minute}'
if not os.path.isdir("models"):
    os.mkdir("models")
config.log_save_dir = f'{config.log_root}/{config.save_dir}_{idx}'
config.save_dir = f'models/{config.save_dir}_{idx}'
if not os.path.isdir(config.save_dir):
    os.mkdir(config.save_dir)
if not os.path.isdir(config.log_root):
    os.mkdir(config.log_root)
if not os.path.isdir(config.log_save_dir):
    os.mkdir(config.log_save_dir)

# setting actual file path
data_basedir = f"{config.data}/" if not config.test else f"data/try_out/{config.data}/"
config.train_data_path = data_basedir + "train.txt"
config.eval_data_path = data_basedir + "val.txt"
config.decode_data_path = data_basedir + "test.txt"
config.vocab_path = data_basedir.replace("/try_out", "") + "vocab"
