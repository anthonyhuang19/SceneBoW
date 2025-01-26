import numpy as np
import shutil
import matplotlib.pyplot as plt
from PIL import Image
import os

def visualize(
    CATEGORIES,
    test_image_paths,
    test_labels_ids,
    predicted_categories_ids,
    train_image_paths,
    train_labels_ids,
):
    results_path = "results"
    visualization_path = "results/visualization.png"
    markdown_path = "results/visualization.md"
    if not os.path.exists(results_path):
        os.mkdir(results_path)
    with open(markdown_path, 'w') as panel:
        panel.write("## Visualization\n")
        panel.write("| Category name | Sample training images | Sample true positives | False positives with true label | False negatives with wrong predicted label |\n")
        panel.write("| :-----------: | :--------------------: | :-------------------: | :-----------------------------: | :----------------------------------------: |\n")

        for k, name in enumerate(CATEGORIES):
            train_image = None
            tp_image = None
            fp_image = None
            fn_image = None
            train_category_path = os.path.join("train", name)
            if os.path.exists(train_category_path):
                train_image = os.path.join(train_category_path, os.listdir(train_category_path)[0])  # Take first image
            test_category_path = os.path.join("test", name)
            if os.path.exists(test_category_path):
                test_ids = np.where(np.array(test_labels_ids) == k)[0].tolist()
                test_images = [test_image_paths[i] for i in test_ids]
                true_labels = [test_labels_ids[i] for i in test_ids]
                predicted_labels = [predicted_categories_ids[i] for i in test_ids]
                for i, (true_label, pred_label) in enumerate(zip(true_labels, predicted_labels)):
                    if true_label == pred_label:
                        if tp_image is None:
                            tp_image = test_images[i] 
                    else:
                        if fn_image is None:
                            fn_image = test_images[i] 

            if os.path.exists(test_category_path):
                fp_ids = np.where(np.array(predicted_categories_ids) == k)[0].tolist()
                fp_images = [test_image_paths[i] for i in fp_ids]
                fp_true_labels = [test_labels_ids[i] for i in fp_ids]

                for i, fp_true in enumerate(fp_true_labels):
                    if fp_true != k and fp_image is None:
                        fp_image = fp_images[i]

            images = [train_image, tp_image, fp_image, fn_image]
            images = [img for img in images if img is not None] 
            if len(images) == 4:
                for img_idx, img in enumerate(images):
                    img_name = f"{name}_{['train', 'tp', 'fp', 'fn'][img_idx]}.png"
                    img_path = os.path.join(results_path, img_name)
                    img = Image.open(img)
                    img.save(img_path)
                panel.write(f"| {name} | ![]({name}_train.png) | ![]({name}_tp.png) | ![]({name}_fp.png) | ![]({name}_fn.png) |\n")
    print(f"Markdown table saved as {markdown_path}")
