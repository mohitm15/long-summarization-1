import os
import time
import torch
from torch import nn
from model import Summarizer
from optim import Optimizer
from config import config
from pre_process import DataLoader, batchify, Vocab
from utils import update_loss

def get_model(model_file_path=None, eval=False):
    model = Summarizer()
    optimizer = Optimizer(config.optim, config.lr_coverage if config.cov else config.lr,
                          acc=config.adagrad_init_acc, max_grad_norm=config.max_grad_norm)
    optimizer.set_parameters(model.parameters())

    step, loss = 1, 0
    if model_file_path is not None:
        checkpoint = torch.load(model_file_path)
        step = checkpoint['step']
        loss = checkpoint['loss']

        model_state_dict = dict([(k, v) for k, v in checkpoint['model'].items()])
        model.load_state_dict(model_state_dict, strict=False)

        if not config.cov and not eval:
            optimizer.optim.load_state_dict(checkpoint['optimizer'])
            if config.cuda:
                for state in optimizer.optim.state.values():
                    for k, v in checkpoint.items():
                        if torch.is_tensor(v):
                            state[k] = v.cuda()
    if config.cuda:
        model = model.cuda()
        optimizer.set_parameters(model.parameters())
    return model, optimizer, step, loss


def trainEpochs(epochs, data, vocab, model_save_dir, model_file_path=None):
    def get_train_batches():
        return batchify(data.get_training_examples(), config.batch_size, vocab)

    model, optim, step, loss = get_model(model_file_path)
    start = time.time()

    for ep in range(epochs):
        batches = get_train_batches()
        for batch in batches:
            optim.zero_grad()
            loss_batch, pred = model(batch)
            loss_batch.backward()
            optim.step()
            loss_batch = loss_batch.item()
            loss = update_loss(loss_batch, loss)
            if step % config.print_interval == 0:
                time_took = time.time()-start
                start = time.time()
                print(f'epoch {ep} ({step} steps); loss: {loss:.4f}, time: {time_took:.2f}s ({time_took/config.print_interval:.1f} step/s)')
                try:
                    print("output: "+" ".join([vocab.get(x, batch.articles[0].oovv.get(x, " ")) for x in pred]))
                    print(f"target: {' '.join(batch.abstracts[0].words)}")
                except:
                    pass
            if step % config.save_interval == 0:
                checkpoint = {
                    'config': config,
                    'epoch': ep,
                    'step': step,
                    'loss': loss,
                    'optimizer': optim.optim.state_dict(),
                    'model': model.module.state_dict() if len(config.gpus) > 1 else model.state_dict()
                }
                model_save_path = os.path.join(model_save_dir, 'model_%d_%d' % (step, int(time.time())))
                torch.save(checkpoint, model_save_path)
            step += 1

def main():
    data = DataLoader(config)
    vocab = data.vocab
    trainEpochs(config.ep, data, vocab, config.save_dir, config.train_from)

if __name__ == '__main__':
    main()
