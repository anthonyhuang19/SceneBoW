# Loading dataset and convert it into vector image
import os


def getting_categories():
    main_folder = "train"
    folders = [
        f
        for f in os.listdir(main_folder)
        if os.path.isdir(os.path.join(main_folder, f))
    ]
    folders.sort(key=lambda folder: folder[0])
    return folders


def load_dataset(categories=getting_categories()):

    train_image_path = []
    testing_image_path = []
    testing_label=[]
    training_label=[]
    for cat in categories:
        path_train = "train/" + cat
        path_test = "test/" + cat
        
        training_images = [
            str("train/" + cat + "/" + f)
            for f in os.listdir(path_train)
            if f.lower().endswith((".jpg", ".jpeg"))
        ]
        for _ in range(len(training_images)):
            training_label.append(cat)
        
        testing_images = [
            str("test/" + cat + "/" + f)
            for f in os.listdir(path_test)
            if f.lower().endswith((".jpg", ".jpeg"))
        ]
        for _ in range(len(testing_images)):
            testing_label.append(cat)
        
        train_image_path.extend(training_images)
        testing_image_path.extend(testing_images)
    return train_image_path, testing_image_path,training_label,testing_label


# if __name__ == '__main__':
#     train_image_path, testing_image_path,train_label,test_label = load_dataset()
#     print(len(train_label)," ",len(train_image_path))
