from load_image import *
import numpy as np
from sklearn.metrics import classification_report
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix

def calculate_accuracy(y_true, y_pred):
    y_true = np.array(y_true)
    y_pred = np.array(y_pred)
    correct_predictions = np.sum(y_true == y_pred)
    total_predictions = len(y_true)
    accuracy = correct_predictions / total_predictions
    return accuracy

def evaluate_classifier(true_labels, predicted_labels):
    print(classification_report(true_labels, predicted_labels))


def build_confusion_mtx(true_labels, predicted_labels, category_names, save_image=False, image_path="confusion_matrix.png"):
    
    cm = confusion_matrix(true_labels, predicted_labels)
    np.set_printoptions(precision=2)
    cm_normalized = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
    plt.figure()
    plot_confusion_matrix(cm_normalized, category_names, title='Normalized Confusion Matrix')
    if save_image:
        plt.savefig(image_path)

def plot_confusion_matrix(cm, category_names, title='Confusion matrix', cmap=plt.cm.Blues):
    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title)
    plt.colorbar()
    
    tick_marks = np.arange(len(category_names))
    plt.xticks(tick_marks, category_names, rotation=45)
    plt.yticks(tick_marks, category_names)
    
    plt.tight_layout()
    plt.ylabel('True label')
    plt.xlabel('Predicted label')

def evaluation_result(test_label, predicted_categories,k,method):
    #first evaluation accuracy
    acc1 = calculate_accuracy(test_label, predicted_categories)
    print(f"Accuracy with k={k}: {acc1 * 100:.2f}%")
    #second evaluation classifier
    # evaluate_classifier(test_label, predicted_categories)
    #Draw confusion matrix
    build_confusion_mtx(test_label, predicted_categories, getting_categories(), save_image=True, image_path="result/"+method+str(k)+".jpg")
    return acc1