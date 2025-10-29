import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.lines as mlines
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay

def eval_history(args, histories, output_dir):
    col = ['tab:blue', 'tab:orange', 'tab:green', 'tab:red', 'tab:purple']
    unit = args.epochs / 10
    fig, ax = plt.subplots(1, 2, figsize=(20, 6))
    for i, (fold, record) in enumerate(histories.items(), start=0):
        history = record['Fold']
        ax[0].plot(history[:, 0], history[:, 1], label=f'Fold : {fold+1}', color=col[i % len(col)])
        ax[0].plot(history[:, 0], history[:, 2], color=col[i % len(col)], linestyle=':')
        ax[1].plot(history[:, 0], history[:, 3], label=f'Fold : {fold+1}', color=col[i % len(col)])
        ax[1].plot(history[:, 0], history[:, 4], color=col[i % len(col)], linestyle=':')
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

def confusion(args, labels, preds, classes, output_dir):
    col_dict = {
                'Top': 'Reds',
                'Side': 'Blues',
                'Fusion': 'Greens'
            }
    cm = confusion_matrix(labels, preds, labels = list(range(len(classes))))
    disp = ConfusionMatrixDisplay(confusion_matrix = cm, display_labels = classes)
    _,ax = plt.subplots(figsize = (8,8))
    disp.plot(cmap=col_dict.get(args.view), xticks_rotation = 90, ax = ax, colorbar=False)
    for t in ax.texts:
        t.set_fontsize(11)
        if t.get_text() == '0':
            t.set_text('-')
    ax.set_xlabel('Predicted Label',fontsize = 16)
    ax.set_ylabel('True Label',fontsize = 16)
    ax.set_title(f'{args.view} : {args.model}',fontsize = 24)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir,'confusion_matrix.png'))
    plt.close()

def summarize_result(args, loss, acc, f1, kappa):
    row = {
        'View': args.view,
        'Model': args.model,
        'Loss': f'{np.mean(loss):.4f} ± {np.std(loss):.4f}',
        'Acc': f'{np.mean(acc):.4f} ± {np.std(acc):.4f}',
        'macro F1': f'{np.mean(f1):.4f} ± {np.std(f1):.4f}',
        'Kappa': f'{np.mean(kappa):.4f} ± {np.std(kappa):.4f}'
    }
    if os.path.exists('Output/Results.csv'):
        df = pd.read_csv('Output/Results.csv', dtype=str)
        mask = (df['View'] == args.view) & (df['Model'] == args.model)
        if mask.any():
            df.loc[mask, ['Loss', 'Acc', 'macro F1', 'Kappa']] = row['Loss'], row['Acc'], row['macro F1'], row['Kappa']
        else:
            df = pd.concat([df, pd.DataFrame([row])], ignore_index=True)
    else:
        df = pd.DataFrame([row])
        df.to_csv('Output/Results.csv', index=False)

    view_order = ['Top', 'Side', 'Fusion']
    model_order = ['ResNet18', 'ViT', 'HCTNet', 'MidNet']
    df['View'] = pd.Categorical(df['View'], categories=view_order, ordered=True)
    df['Model'] = pd.Categorical(df['Model'], categories=model_order, ordered=True)
    df = df.sort_values(['View', 'Model']).reset_index(drop=True)
    df.to_csv('Output/Results.csv', index=False)