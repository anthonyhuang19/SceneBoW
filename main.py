from load_image import *
from knn import *
from tiny_image import *
from evaluation import *
from vocab_build import *
from create_bag_sift import *
from visualization import *
import sys
import matplotlib.pyplot as plt
import numpy as np


# producing comparison table part :6
def save_table_as_image(k_values, accuracy_1, filename="comparison_table.png"):
    columns = ["k", "Accuracy 1"]
    data = np.array([k_values, accuracy_1]).T

    fig, ax = plt.subplots(figsize=(8, 4))
    ax.axis("off")
    table = ax.table(
        cellText=data,
        colLabels=columns,
        loc="center",
        cellLoc="center",
        bbox=[0, 0, 1, 1],
    )

    table.auto_set_font_size(False)
    table.set_fontsize(12)
    table.scale(1.2, 1.2)

    plt.savefig(filename, bbox_inches="tight", dpi=300)
    plt.show()

    print(f"Table saved as '{filename}'.")


def main(method, k=50):
    train_image_path, testing_image_path, train_label, test_label = load_dataset()

    # Convert image into tiny image or bag of sift
    if method == "tiny_image":

        k = k
        train_dataset = tiny_image(train_image_path)
        test_dataset = tiny_image(testing_image_path)
    elif method == "bag_word":
        k = k
        kmeans = vocab_build(train_image_path, vocab_size=k)
        train_dataset = create_bag_of_sifts(train_image_path, kmeans)
        test_dataset = create_bag_of_sifts(testing_image_path, kmeans)

    # Classifier method:
    # part 5:Nearest neighbor classifier: You should now measure how well your bag of SIFT representation works when paired with a nearest neighbor classifier.
    predicted_categories = knn_clasification(
        train_image=train_dataset, train_label=train_label, test_image=test_dataset
    )
    with open("predicted_categories.txt", "w") as file:
        for category in predicted_categories:
            file.write(category + "\n")
    # Evaluation
    #acc1 = evaluation_result(test_label, predicted_categories, k, method)
    
    #Visulization for image
    print("Visualization")
    categories = getting_categories()
    cat_into_id = {v: k for k, v in enumerate(categories)}
    test_labels_id = [cat_into_id[x] for x in test_label]
    predicted_categories_id = [cat_into_id[x] for x in predicted_categories]
    train_labels_id = [cat_into_id[x] for x in train_label]
    visualize(
        getting_categories(),
        testing_image_path,
        test_labels_id,
        predicted_categories_id,
        train_image_path,
        train_labels_id,
    )
    #return acc1

#camperison for bag_word with change k parameters
def comparison(k=[10, 20, 30, 40, 50, 60, 70, 80, 90, 100]):
    accuracy_1 = []
    for value_k in k:
        (acc1) = main(method=None, k=value_k)
        accuracy_1.append(acc1)
    save_table_as_image(k, accuracy_1)
    print("Finishing produce comparison table")


if __name__ == "__main__":

    if len(sys.argv) != 3:
        print("Usage: python3 main.py <method> <type>")
        sys.exit(1)

    method = sys.argv[1]
    type = int(sys.argv[2])
    print(type, " ", method)
    if type == 1:
        main(method)
    else:
        comparison()
