import argparse

def get_args():
    parser = argparse.ArgumentParser('Multimodal Classification')
    parser.add_argument('--size', default=224, type=int)
    parser.add_argument('--splits',default=5,type=int)
    parser.add_argument('--epochs',default=20,type=int)
    parser.add_argument('--lr',default=1e-3,type=float)
    parser.add_argument('--weight_decay',default=1e-2,type=float)
    parser.add_argument('--batch_size',default=32,type=int)
    parser.add_argument('--view',default='Top',choices=['Top','Side','Fusion'],type = str)
    parser.add_argument('--model',default='ResNet18',choices = ['ResNet50','ViT','SwinT','Proposed'],type = str)
    parser.add_argument('--format',default='{l_bar}{bar:60} | {n_fmt:>2} / {total_fmt:>2} [{elapsed} < {remaining} , {rate_fmt}] {postfix}',type = str)
    parser.add_argument('--ascii', default='-▮', type=str)
    return parser.parse_args()