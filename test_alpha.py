import warnings
warnings.simplefilter("ignore")

import os
import matplotlib.pyplot as plt
import matplotlib.lines as mlines
import logging
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from collections import OrderedDict
from torchvision import transforms
from torch.utils.data import DataLoader
from torch.optim.lr_scheduler import CosineAnnealingLR
from sklearn.model_selection import train_test_split
from sklearn.metrics import f1_score, cohen_kappa_score
from tqdm import tqdm
from Utils.Image import OriginalDataset
from Utils.seed import get_seed
from Utils.args import get_args
from Utils.eval_func import eval_history

from Models.MidNet import *

for name in logging.root.manager.loggerDict:
    if "huggingface" in name or "timm" in name:
        logging.getLogger(name).setLevel(logging.ERROR)

def train(args, output_dir):
    def make_dataset(idx, imgs, labels):
        selected_imgs = [imgs[i] for i in idx]
        selected_labels = [labels[i] for i in idx]
        return selected_imgs, selected_labels
    def get_transform(train):
        if train:
            transform = transforms.Compose([
                transforms.RandomResizedCrop(args.size, scale=(0.9, 1.0)),
                transforms.ColorJitter(brightness=0.1, contrast=0.1, saturation=0.05, hue=0.02),
                #transforms.RandomAffine(degrees=0, translate=(0.05, 0.05), shear=(-5, 5)),
                transforms.RandomErasing(p=0.1, scale=(0.005, 0.05), ratio=(0.3, 3.3)),
                transforms.Normalize([0.5]*3, [0.5]*3)
                ])
        else:
            transform = transforms.Compose([
                transforms.Normalize([0.5]*3, [0.5]*3)
                ])
        return transform

    get_seed(42)
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

    label_df = pd.read_csv('Data/Labels_Count.csv')
    classes = tuple(label_df.sort_values('Encode')['Label'])
    df = torch.load('Data/Dataset.pt', weights_only=False)
    imgs, labels = df[args.view], df['Label']

    np_labels = labels.numpy()
    df_train, df_test, label_train, label_test = train_test_split(imgs, labels, test_size=0.2, random_state=42, stratify=np_labels)
    train_dataset = OriginalDataset(args, imgs=df_train, imgs_other=None, labels=label_train, transform=get_transform(True))
    test_dataset = OriginalDataset(args, imgs=df_test, imgs_other=None, labels=label_test, transform=get_transform(False))
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False)

    counts = label_df.sort_values('Encode')['Counts'].to_numpy()

    class_weights = 1.0 / np.sqrt(np.log1p(counts))
    class_weights = class_weights / class_weights.max()
    class_weights = torch.tensor(class_weights, dtype=torch.float32).to(device)

    log_path = os.path.join(output_dir, 'train.log')
    logging.basicConfig(filename=log_path, level=logging.INFO, format='%(message)s', filemode='w')

    alpha_list = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]
    histories = {}
    n_alpha = 0
    with tqdm(total=len(alpha_list), desc=f'{f"alpha":<10}', bar_format=args.format, ascii=args.ascii) as pbar:
        for alpha in alpha_list:
            model = MidNet(n_classes=len(classes)).to(device)
            criterion = nn.CrossEntropyLoss(weight=class_weights, label_smoothing=0.1)
            optimizer = optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
            scheduler = CosineAnnealingLR(optimizer, T_max=args.epochs, eta_min=1e-5)
            history = np.zeros((0,5))
            logging.info(f'[alpha : {alpha}]')
            with tqdm(total=args.epochs, desc=f'{f"Epoch X":<10}', bar_format=args.format, ascii=args.ascii, leave=False) as qbar:
                for epoch in range(1, args.epochs + 1):
                    train_loss = test_loss = 0.0
                    train_acc = test_acc = 0.0
                    n_train = n_test = 0
                    Preds, Labels = [], []
                    model.train()
                    for imgs, labels in tqdm(train_loader, desc=f'{"Train":<10}', bar_format=args.format, ascii=args.ascii, leave=False):
                        inputs = (imgs.to(device), )
                        labels = labels.to(device)
                        optimizer.zero_grad()
                        outputs, kd_loss = model(*inputs, kd=True)
                        ce_loss = criterion(outputs, labels)
                        loss = ce_loss + alpha * kd_loss
                        loss.backward()
                        optimizer.step()

                        train_loss += loss.item()
                        predicted = outputs.argmax(dim=1)
                        n_train += labels.size(0)
                        train_acc += (predicted == labels).sum().item()

                    model.eval()
                    with torch.no_grad():
                        for imgs, labels in tqdm(test_loader, desc=f'{"Test":<10}', bar_format=args.format, ascii=args.ascii, leave=False):
                            inputs = (imgs.to(device), )
                            labels = labels.to(device)
                            outputs, kd_loss = model(*inputs, kd=True)
                            ce_loss = criterion(outputs, labels)
                            loss = ce_loss + alpha * kd_loss
                            test_loss += loss.item()
                            predicted = outputs.argmax(dim=1)
                            n_test += labels.size(0)
                            test_acc += (predicted == labels).sum().item()

                            Preds.extend(predicted.cpu())
                            Labels.extend(labels.cpu())

                    train_loss /= n_train
                    train_acc /= n_train
                    test_loss /= n_test
                    test_acc /= n_test

                    eval_preds = np.array(Preds)
                    eval_labels = np.array(Labels)
                    macro_f1 = f1_score(eval_labels, eval_preds, average='macro')
                    kappa = cohen_kappa_score(eval_labels, eval_preds)
                    item = np.array([epoch, train_loss, test_loss, train_acc, test_acc])
                    history = np.vstack((history, item))
                    logging.info(f"[{epoch:>2}/{args.epochs:>2}] (Train) Loss={train_loss:.4f} Acc={train_acc:.4f} (Test) Loss={test_loss:.4f} Acc={test_acc:.4f} F1={macro_f1:.4f} Kappa={kappa:.4f}")
                    scheduler.step()
                    qbar.set_postfix(OrderedDict([
                        ('Train', f'{train_loss:.4f}'),
                        ('Test', f'{test_loss:.4f}')
                    ]))
                    qbar.update()
            histories[n_alpha] = {'alpha': history}
            n_alpha += 1

            row = {
                'Alpha': float(alpha),
                'Loss': f'{test_loss:.4f}',
                'Acc': f'{test_acc:.4f}',
                'macro F1': f'{macro_f1:.4f}',
                'Kappa': f'{kappa:.4f}',
            }
            filename = f'{output_dir}/Comparison.csv'
            if os.path.exists(filename):
                df = pd.read_csv(filename)
                df['Alpha'] = df['Alpha'].astype(float)
                mask = (df['Alpha'] == alpha)
                if mask.any():
                    df.loc[mask, ['Loss', 'Acc', 'macro F1', 'Kappa']] = row['Loss'], row['Acc'], row['macro F1'], row['Kappa']
                else:
                    df = pd.concat([df, pd.DataFrame([row], columns=df.columns)], ignore_index=True)
            else:
                df = pd.DataFrame([row])

            alpha_order = [0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]
            df['Alpha'] = pd.Categorical(df['Alpha'], categories=alpha_order, ordered=True)
            df = df.sort_values(['Alpha']).reset_index(drop=True)
            df.to_csv(filename, index=False)

            cmap = plt.get_cmap('tab10')
            unit = args.epochs / 10
            fig, ax = plt.subplots(1, 2, figsize=(20, 6))
            for i, (n, record) in enumerate(histories.items(), start=0):
                col = cmap(i % 10)
                history = record['alpha']
                ax[0].plot(history[:, 0], history[:, 1], label=rf'$\alpha$ = {alpha_list[i]}', color=col)
                ax[0].plot(history[:, 0], history[:, 2], color=col, linestyle=':')
                ax[1].plot(history[:, 0], history[:, 3], label=rf'$\alpha$ = {alpha_list[i]}', color=col)
                ax[1].plot(history[:, 0], history[:, 4], color=col, linestyle=':')
            ax[0].set_title('Loss Curve', fontsize=20)
            ax[0].set_xlabel('Epochs', fontsize=14)
            ax[0].set_ylabel('Loss', fontsize=14)
            ax[0].set_xticks(np.arange(0, args.epochs + 1, unit))
            ax[0].grid(alpha=0.7, linestyle=':')
            ax[0].legend()
            ax[1].set_title('Accuracy Curve', fontsize=20)
            ax[1].set_xlabel('Epochs', fontsize=14)
            ax[1].set_ylabel('Accuracy', fontsize=14)
            ax[1].set_xticks(np.arange(0, args.epochs + 1, unit))
            ax[1].grid(alpha=0.7, linestyle=':')
            ax[1].legend()
            train_line = mlines.Line2D([], [], color='black', label='Train', linestyle='-')
            test_line = mlines.Line2D([], [], color='black', label='Test', linestyle=':')
            fig.legend(handles=[train_line, test_line],
                        loc='lower center', bbox_to_anchor=(0.5, -0.05),
                        ncol=2, fontsize=12)
            fig.subplots_adjust(bottom=0.1)
            plt.savefig(os.path.join(output_dir, 'leaning_curve.png'), bbox_inches='tight')
            plt.close()

            pbar.update()

if __name__ == '__main__':
    args = get_args()
    output_dir = f'Output/{args.view}/{args.model}/alpha'
    if not os.path.exists(output_dir):
        os.makedirs(output_dir, exist_ok=True)
    print(f'[{args.view:^10}:{args.model:^10}]')
    train(args, output_dir)