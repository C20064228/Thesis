import os
import warnings
import time
import logging
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.lines as mlines
import torch
import torch.nn as nn
import torch.optim as optim
from collections import OrderedDict
from torchvision import transforms
from torchvision.transforms import ToPILImage
from torch.utils.data import DataLoader
from torch.optim.lr_scheduler import CosineAnnealingLR
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import f1_score, cohen_kappa_score
from tqdm import tqdm
from Utils.Image import OriginalDataset
from Utils.seed import get_seed
from Utils.args import get_args
from Utils.eval_func import eval_history, confusion, summarize_result

from Models.ResNet50 import *
from Models.ViT import *
from Models.MidNet import *

os.environ["HF_HUB_OFFLINE"] = "1"
os.environ["HF_HUB_HTTP_VERBOSE"] = "0"
logging.getLogger("httpx").disabled = True
logging.getLogger("httpcore").disabled = True
warnings.simplefilter("ignore")
warnings.filterwarnings("ignore", category=FutureWarning, module="torch.cuda")

for name in logging.root.manager.loggerDict:
    if "huggingface" in name or "timm" in name:
        logging.getLogger(name).setLevel(logging.ERROR)

def train(args, output_dir):
    def choose_model(args, classes):
        n_classes = len(classes)
        if args.view in ['Top', 'Side']:
            model_dict = {
                'ResNet18': ResNet18,
                'ViT': ViT,
                'HCTNet': MidNet,
                'MidNet': MidNet
            }
        else:
            model_dict = {
                'ResNet18': ResNet18_F,
                'ViT': ViT_F,
                'HCTNet': MidNet_F,
                'MidNet': MidNet_F
            }
        model_class = model_dict.get(args.model)
        model = model_class(n_classes)
        return model
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
    imgs_top, imgs_side = df['Top'], df['Side']
    labels = df['Label']

    loaders = {}
    skf = StratifiedKFold(n_splits=args.splits, shuffle=True, random_state=42)
    for fold, (train_idx, test_idx) in enumerate(skf.split(imgs_top, labels)):
        if args.view in ['Top', 'Side']:
            imgs = imgs_top if args.view == 'Top' else imgs_side
            df_train, label_train = make_dataset(train_idx, imgs, labels)
            df_test, label_test = make_dataset(test_idx, imgs, labels)
            train_dataset = OriginalDataset(args, imgs=df_train, imgs_other=None, labels=label_train, transform=get_transform(True))
            test_dataset = OriginalDataset(args, imgs=df_test, imgs_other=None, labels=label_test, transform=get_transform(False))
        else:
            top_train, label_train = make_dataset(train_idx, imgs_top, labels)
            side_train, _ = make_dataset(train_idx, imgs_side, labels)
            top_test, label_test = make_dataset(test_idx, imgs_top, labels)
            side_test, _ = make_dataset(test_idx, imgs_side, labels)
            train_dataset = OriginalDataset(args, imgs=top_train, imgs_other=side_train, labels=label_train, transform=get_transform(True))
            test_dataset = OriginalDataset(args, imgs=top_test, imgs_other=side_test, labels=label_test, transform=get_transform(False))
        loaders[fold] = {
            'Train': DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True),
            'Test': DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False)
        }

    counts = label_df.sort_values('Encode')['Counts'].to_numpy()

    class_weights = 1.0 / np.sqrt(np.log1p(counts))
    class_weights = class_weights / class_weights.max()
    class_weights = torch.tensor(class_weights, dtype=torch.float32).to(device)

    log_path = os.path.join(output_dir, 'train.log')
    logging.basicConfig(filename=log_path, level=logging.INFO, format='%(message)s', filemode='w')

    fold_loss, fold_acc, fold_f1, fold_kappa = [], [], [], []
    all_preds, all_labels = [], []
    all_times = []

    histories = {}
    with tqdm(total=args.splits, desc=f'{f"Fold  X":<10}', bar_format=args.format, ascii=args.ascii) as pbar:
        for fold in loaders:
            model = choose_model(args, classes)
            model.to(device)
            criterion = nn.CrossEntropyLoss(weight=class_weights, label_smoothing=0.1)
            optimizer = optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
            #scheduler = StepLR(optimizer, step_size=5, gamma=0.8)
            scheduler = CosineAnnealingLR(optimizer, T_max=args.epochs, eta_min=1e-5)
            history = np.zeros((0,5))
            logging.info(f'[Fold : {fold + 1}]')
            with tqdm(total=args.epochs, desc=f'{f"Epoch X":<10}', bar_format=args.format, ascii=args.ascii, leave=False) as qbar:
                for epoch in range(1, args.epochs + 1):
                    train_loss = test_loss = 0.0
                    train_acc = test_acc = 0.0
                    n_train = n_test = 0
                    Preds, Labels = [], []
                    model.train()
                    for imgs, labels in tqdm(loaders[fold]['Train'], desc=f'{"Train":<10}', bar_format=args.format, ascii=args.ascii, leave=False):
                        labels = labels.to(device)
                        if args.view in ['Top', 'Side']:
                            inputs = (imgs.to(device),)
                        else:
                            imgs_top, imgs_side = imgs
                            inputs = (imgs_top.to(device), imgs_side.to(device))
                        optimizer.zero_grad()
                        if args.model == 'MidNet':
                            outputs, kd_loss = model(*inputs, kd=True)
                            ce_loss = criterion(outputs, labels)
                            loss = ce_loss + args.alpha * kd_loss
                        else:
                            outputs = model(*inputs)
                            loss = criterion(outputs, labels)
                        loss.backward()
                        optimizer.step()

                        train_loss += loss.item()
                        predicted = outputs.argmax(dim=1)
                        n_train += labels.size(0)
                        train_acc += (predicted == labels).sum().item()

                    model.eval()
                    with torch.no_grad():
                        for imgs, labels in tqdm(loaders[fold]['Test'], desc=f'{"Test":<10}', bar_format=args.format, ascii=args.ascii, leave=False):
                            labels = labels.to(device)
                            if args.view in ['Top', 'Side']:
                                inputs = (imgs.to(device),)
                            else:
                                imgs_top, imgs_side = imgs
                                inputs = (imgs_top.to(device), imgs_side.to(device))
                            start_time = time.time()
                            if args.model == 'MidNet':
                                outputs, kd_loss = model(*inputs, kd=True)
                                end_time = time.time()
                                ce_loss = criterion(outputs, labels)
                                loss = ce_loss + args.alpha * kd_loss
                            else:
                                outputs = model(*inputs)
                                end_time = time.time()
                                loss = criterion(outputs, labels)
                            test_loss += loss.item()
                            predicted = outputs.argmax(dim=1)
                            n_test += labels.size(0)
                            test_acc += (predicted == labels).sum().item()
                            diff_time = end_time - start_time
                            all_times.append(diff_time)

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
            fold_loss.append(test_loss)
            fold_acc.append(test_acc)
            fold_f1.append(macro_f1)
            fold_kappa.append(kappa)
            all_labels.extend(Labels)
            all_preds.extend(Preds)
            histories[fold] = {'Fold': history}
            pbar.set_postfix(OrderedDict([
                ('Loss', f'{np.mean(fold_loss):.4f}'),
                ('Acc', f'{np.mean(fold_acc):.4f}')
            ]))
            pbar.update()
            eval_history(args, histories, output_dir)

    eval_history(args, histories, output_dir)
    confusion(args, all_labels, all_preds, classes, output_dir)
    summarize_result(args, fold_loss, fold_acc, fold_f1, fold_kappa, all_times)

if __name__ == '__main__':
    args = get_args()
    output_dir = f'Output/{args.view}/{args.model}'
    if not os.path.exists(output_dir):
        os.makedirs(output_dir, exist_ok=True)
    print(f'[{args.view:^10}:{args.model:^10}]')
    train(args, output_dir)