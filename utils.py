import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix
import os
import numpy as np

def save_fig(fig_id, tight_layout=True, fig_extension="png", resolution=300):
    # create'figures' folder if it doesn't exist
    figures_dir = 'figures'
    if not os.path.exists(figures_dir):
        os.makedirs(figures_dir)
    
    # construct the file path
    path = os.path.join(figures_dir, f"{fig_id}.{fig_extension}")
    if tight_layout:
        plt.tight_layout()
    
    # save the figure to the specified path
    plt.savefig(path, format=fig_extension, dpi=resolution)

def compute_classwise_accuracy(outputs,class_names):
    all_preds, all_labels = [], []
    for output in outputs:
        all_preds.append(output['preds'].cpu().numpy())
        all_labels.append(output['labels'].cpu().numpy())

    all_preds = np.concatenate(all_preds)
    all_labels = np.concatenate(all_labels)

    #per-class accuracy
    class_accuracy = {}
    for class_idx in np.unique(all_labels):
        correct = (all_preds[all_labels == class_idx] == class_idx).sum()
        total = (all_labels == class_idx).sum()
        class_accuracy[class_names[class_idx]] = correct / total

    # Plot class-wise accuracy
    plt.figure(figsize=(8, 5))
    plt.bar(class_accuracy.keys(), class_accuracy.values(), color='skyblue')
    plt.xlabel("Class")
    plt.ylabel("Accuracy")
    plt.title(f"Per-Class Accuracy")
    plt.xticks(rotation=45)
    save_fig("class_wise_acc")
    plt.show()

    #plot confusion matrix
    cm = confusion_matrix(all_labels, all_preds)

    fig, ax = plt.subplots(figsize=(8, 6))
    cax = ax.matshow(cm, cmap="Blues")
    plt.colorbar(cax)

    # Set axis labels
    ax.set_xticks(np.arange(len(class_names)))
    ax.set_yticks(np.arange(len(class_names)))
    ax.set_xticklabels(class_names, rotation=45, ha="right")
    ax.set_yticklabels(class_names)

    # Label each cell with its count
    for i in range(len(class_names)):
        for j in range(len(class_names)):
            ax.text(j, i, str(cm[i, j]), va='center', ha='center', color='black')

    plt.xlabel("Predicted")
    plt.ylabel("Actual")
    plt.title(f"Confusion Matrix")
    save_fig("confusion_matrix")
    plt.show()

    return class_accuracy
