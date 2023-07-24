import argparse

import torch
from torch.utils.data import DataLoader

from data import LSCNNDataset
from loss import LSCLoss, NTXentLoss
from model.lscnn import LSCNN
from utils import load_json, EarlyStopping


def main():
    min_val_loss = 10000
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    hparams = load_json('./configs', 'configs_lscnn')
    early_stopping = EarlyStopping(patience=15)

    # loading
    print('Loading data...')
    train_set = LSCNNDataset(dataset_type='train', seed=hparams['seed'])
    val_set = LSCNNDataset(dataset_type='val', seed=hparams['seed'])
    print(len(train_set))
    print(len(val_set))
    train_loader = DataLoader(train_set, **hparams['train_loading'])
    val_loader = DataLoader(val_set, **hparams['val_loading'])

    model = LSCNN(hparams['is_project'])
    model = model.to(device)
    optim = torch.optim.RMSprop(model.parameters(), **hparams['optim'])

    criterion_mse = torch.nn.MSELoss()
    if hparams['lsc_loss'] == 'ed':
        criterion_lsc = LSCLoss()
    elif hparams['lsc_loss'] == 'cl':
        criterion_lsc = NTXentLoss(device='cuda', batch_size=16,
                                   temperature=0.1, use_cosine_similarity=True)

    # train
    for epoch in range(hparams['epochs']):
        model.train()
        train_loss = []
        pd_loss = []
        fd_loss = []
        lsc_loss = []
        for inputs, labels in train_loader:
            inputs = inputs.to(device)
            labels = labels.to(device)
            pd_output, fd_output, pd_latent_rep, fd_latent_rep = model(inputs)

            loss = criterion_mse(pd_output, labels) + criterion_mse(fd_output, labels)\
                   + hparams['lambda_lsc'] * criterion_lsc(pd_latent_rep, fd_latent_rep)

            optim.zero_grad()
            loss.backward()
            optim.step()
            train_loss.append(loss.item())
            pd_loss.append(criterion_mse(pd_output, labels).item())
            fd_loss.append(criterion_mse(fd_output, labels).item())
            lsc_loss.append(criterion_lsc(pd_latent_rep, fd_latent_rep).item())

        with torch.no_grad():
            model.eval()
            val_loss = []
            val_lsc_loss = []
            for inputs, labels in val_loader:
                inputs = inputs.to(device)
                labels = labels.to(device)
                pd_output, fd_output, pd_latent_rep, fd_latent_rep = model(inputs)

                loss = criterion_mse(pd_output, labels) + criterion_mse(fd_output, labels) \
                       + hparams['lambda_lsc'] * criterion_lsc(pd_latent_rep, fd_latent_rep)

                val_loss.append(loss.item())
                val_lsc_loss.append(criterion_lsc(pd_latent_rep, fd_latent_rep).item())

        # early stopping
        early_stopping(sum(val_loss) / len(val_loss))
        if early_stopping.early_stop:
            print("Early stopping")
            break
        if sum(val_loss) / len(val_loss) < min_val_loss:
            min_val_loss = sum(val_loss) / len(val_loss)
            torch.save(model.state_dict(), "./checkpoints/best_model.pth")
            print("saved!")

        print("Epoch: {}/{}    Train Loss: {}    Val Loss: {}".format(epoch, hparams['epochs'], sum(train_loss) / len(train_loss),
                                                                      sum(val_loss) / len(val_loss)))


if __name__ == '__main__':
    main()
