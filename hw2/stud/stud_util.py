import matplotlib.pyplot as plt
import torch
from gensim.models import KeyedVectors
import numpy as np
from sklearn.metrics import confusion_matrix
from sklearn.metrics import ConfusionMatrixDisplay
import pandas as pd
import seaborn as sns



def load_torch_embedding_layer(weights: KeyedVectors, padding_idx: int = 0, freeze: bool = False):
    vectors = weights
    # random vector for pad
    pad = np.random.rand(1, vectors.shape[1])
    print(pad.shape)
    # mean vector for unknowns
    unk = np.mean(vectors, axis=0, keepdims=True)
    print(unk.shape)
    # concatenate pad and unk vectors on top of pre-trained weights
    vectors = np.concatenate((pad, unk, vectors))
    # convert to pytorch tensor
    vectors = torch.FloatTensor(vectors)
    # and return the embedding layer
    return torch.nn.Embedding.from_pretrained(vectors, padding_idx=padding_idx, freeze=freeze)

def get_mask(batch_tensor):
    mask = batch_tensor.eq(0)
    mask = mask.eq(0)
    return mask

def build_pretrain_embedding(embed, word_vocab, embedd_dim=100):
    vocab_size = len(word_vocab.id2word)
    scale = np.sqrt(3.0 / embedd_dim)
    pretrain_emb = np.empty([vocab_size, embedd_dim])
    perfect_match = 0
    case_match = 0
    not_match = 0
    for word, index in word_vocab.word2id.items():
        if word in embed:
            pretrain_emb[index, :] = embed[word]
            perfect_match += 1
        elif word.lower() in embed:
            pretrain_emb[index, :] = embed[word.lower()]
            case_match += 1
        else:
            pretrain_emb[index, :] = np.random.uniform(-scale, scale, [1, embedd_dim])
            not_match += 1

    pretrain_emb[0, :] = np.zeros((1, embedd_dim))
    pretrained_size = len(embed)
    print("Embedding:\n     pretrain word:%s, prefect match:%s, case_match:%s, oov:%s, oov%%:%s" % (
        pretrained_size, perfect_match, case_match, not_match, (not_match + 0.) / vocab_size))
    return pretrain_emb

def create_conf_matrix(y_true, y_pred):
    pred_set = []
    label_set = []
    for id in y_true:
        true = y_true[id]['roles']
        pred = y_pred[id]['roles']
        #reversed_pred = dict(reversed(list(pred.items())))
        pred_index_true = true.keys()
        for pred_id in pred_index_true:
            for i in range(len(pred[pred_id])):
                if (pred[pred_id][i] == '_' or true[pred_id][i] == '_'):
                    continue
                pred_set.append(pred[pred_id][i])
                label_set.append(true[pred_id][i])
    name_label = list(set(label_set) | (set(pred_set)))
    cm = confusion_matrix(label_set, pred_set, normalize='true')
    cm_df = pd.DataFrame(cm, index=name_label, columns=name_label)
    plt.figure(figsize=(27, 27), dpi=200)
    sns.heatmap(cm_df, annot=True, cmap='coolwarm')
    plt.ylabel('True label')
    plt.xlabel('Predicted label')
    plt.title('Normalized Confusion Matrix of SRL homework 2')
    plt.savefig('conf_matrix.png')

    return cm, name_label

def plot_class_dist(values, labels):
    # Figure Size
    fig, ax = plt.subplots(figsize=(16, 9), dpi=100)

    # Horizontal Bar Plot
    ax.barh(list(labels), list(values))

    # Remove axes splines
    for s in ['top', 'bottom', 'left', 'right']:
        ax.spines[s].set_visible(False)

    # Remove x, y Ticks
    ax.xaxis.set_ticks_position('none')
    ax.yaxis.set_ticks_position('none')

    # Add padding between axes and labels
    ax.xaxis.set_tick_params(pad=5)
    ax.yaxis.set_tick_params(pad=10)

    # Add x, y gridlines
    ax.grid(b=True, color='grey',
            linestyle='-.', linewidth=0.5,
            alpha=0.2)

    # Show top values
    ax.invert_yaxis()

    # Add annotation to bars
    for i in ax.patches:
        plt.text(i.get_width() + 0.2, i.get_y() + 0.5,
                 str(round((i.get_width()), 2)),
                 fontsize=10, fontweight='bold',
                 color='grey')

    # Add Plot Title
    ax.set_title('Class distribution of the training set',
                 loc='left', )

    # Show Plot
    plt.savefig('class_dist.png')
