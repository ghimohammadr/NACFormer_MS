import argparse
import torch.nn.functional as F
import torch
from torch import tensor
from Models import APPNPTransformerBlock, fastAPPNPTransformerBlock
import numpy as np
import time
from utils import load_data, coarsening, getData
from sklearn.metrics import accuracy_score, f1_score
import os

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', type=str, default='cora') # options: [cora, citeseer, pubmed, dblp]
    parser.add_argument('--experiment', type=str, default='fullsupervised') # options: [fullsupervised, semisupervised]
    parser.add_argument('--runs', type=int, default=1)
    parser.add_argument('--hidden', type=int, default=64)
    parser.add_argument('--n_heads', type=int, default=8)
    parser.add_argument('--epochs', type=int, default=80)
    parser.add_argument('--early_stopping', type=int, default=10)
    parser.add_argument('--lr', type=float, default=0.01)
    parser.add_argument('--weight_decay', type=float, default=0.0005)
    parser.add_argument('--normalize_features', type=bool, default=True)
    parser.add_argument('--coarsening_ratio', type=float, default=[0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]) 
    parser.add_argument('--coarsening_method', type=str, default='variation_neighborhoods')
    args = parser.parse_args()
    path = "params/"
    if not os.path.isdir(path):
        os.mkdir(path)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    all_acc = []
    all_macro = []

    start = time.time()
    for i in range(args.runs):

        print("Run: ", i)

        data, args.num_features, args.num_classes = getData(args.dataset)
        model = fastAPPNPTransformerBlock(args).to(device)
        optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
        model.reset_parameters()

        best_val_loss = float('inf')
        for ratio in args.coarsening_ratio:

            candidate, C_list, Gc_list = coarsening(data, 1-ratio, args.coarsening_method)
            data, coarsen_features, coarsen_train_labels, coarsen_train_mask, coarsen_val_labels, coarsen_val_mask, coarsen_edge = load_data(
                args.dataset, candidate, C_list, Gc_list, args.experiment)
            data = data.to(device)
            coarsen_features = coarsen_features.to(device)
            coarsen_train_labels = coarsen_train_labels.to(device)
            coarsen_train_mask = coarsen_train_mask.to(device)
            coarsen_val_labels = coarsen_val_labels.to(device)
            coarsen_val_mask = coarsen_val_mask.to(device)
            coarsen_edge = coarsen_edge.to(device)

            if args.normalize_features:
                coarsen_features = F.normalize(coarsen_features, p=2)
                data.x = F.normalize(data.x, p=2)

            val_loss_history = []
            for epoch in range(args.epochs):

                model.train()
                optimizer.zero_grad()
                out, _ = model(coarsen_features, coarsen_edge)
                loss = F.nll_loss(out[coarsen_train_mask], coarsen_train_labels[coarsen_train_mask])
                loss.backward()
                optimizer.step()

                model.eval()
                pred, _ = model(coarsen_features, coarsen_edge)
                val_loss = F.nll_loss(pred[coarsen_val_mask], coarsen_val_labels[coarsen_val_mask]).item()

                if val_loss < best_val_loss and epoch > args.epochs // 2:
                    best_val_loss = val_loss
                    torch.save(model.state_dict(), path + 'checkpoint-best-acc.pkl')

                val_loss_history.append(val_loss)
                if args.early_stopping > 0 and epoch > args.epochs // 2:
                    tmp = tensor(val_loss_history[-(args.early_stopping + 1):-1])
                    if val_loss > tmp.mean().item():
                        break

        model.load_state_dict(torch.load(path + 'checkpoint-best-acc.pkl'))
        model.eval()
        pred, logits = model(data.x, data.edge_index)
        test_acc = accuracy_score(data.y[data.test_mask], logits.argmax(1).cpu().detach().numpy()[data.test_mask])
        f1macro = f1_score(data.y[data.test_mask], logits.argmax(1).cpu().detach().numpy()[data.test_mask], average='macro')
        print("Accuracy and F1 Macro are: ", test_acc, f1macro)
        all_acc.append(test_acc)
        all_macro.append(f1macro)

    end = time.time()
    print('ave_acc: {:.4f}'.format(np.mean(all_acc)), '+/- {:.4f}'.format(np.std(all_acc)))
    print('ave_macro: {:.4f}'.format(np.mean(all_macro)), '+/- {:.4f}'.format(np.std(all_macro)))
    print('ave_time:', (end-start)/args.runs)

