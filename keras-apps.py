# =================================
# ======= import libraries ========
# =================================

from tensorflow.keras.metrics import binary_accuracy, categorical_accuracy, top_k_categorical_accuracy
from tensorflow.keras.models import Model, load_model
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.applications import (
    Xception,
    VGG16,
    VGG19,
    ResNet50,
    ResNet50V2,
    ResNet101,
    ResNet101V2,
    ResNet152,
    ResNet152V2,
    InceptionV3,
    InceptionResNetV2,
    MobileNet,
    MobileNetV2,
    DenseNet121,
    DenseNet169,
    DenseNet201,
    EfficientNetB0,
    EfficientNetB1,
    EfficientNetB2,
    EfficientNetB3,
    EfficientNetB4,
    EfficientNetB5,
    EfficientNetB6,
    EfficientNetB7,
    EfficientNetV2B0,
    EfficientNetV2B1,
    EfficientNetV2B2,
    EfficientNetV2B3,
    EfficientNetV2S,
    EfficientNetV2M,
    EfficientNetV2L,
)
from tensorflow.keras.layers import Dense, Dropout, Flatten
from tensorflow.keras.optimizers import Adam, SGD

from sklearn.preprocessing import LabelBinarizer
from sklearn import metrics

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib import cycler
import seaborn as sns

from PIL import Image
from tqdm import tqdm
import pickle
import splitfolders
import argparse
import math
import os

# =================================
# ========== arguments ============
# =================================

parser = argparse.ArgumentParser()
# experiment name
parser.add_argument("-en", "--exp-name", help="experiment name (default: mobile-net-v2-exp)", default="mobile-net-v2-exp", type=str)
# data
parser.add_argument("-d", "--data-path", help="data path (default: ./data)", default="./data", type=str)
parser.add_argument("-sd", "--splitted-data-path", help="splitted data path (default: ./splitted_data)", default="./splitted_data", type=str)
parser.add_argument("-tr", "--training-ratio", help="training ratio (default: 0.8)", default=0.8, type=float)
# run settings
parser.add_argument("-t", "--train", help="train (default: False)", default=False, action="store_true")
parser.add_argument("-e", "--eval", help="eval (default: False)", default=False, action="store_true")
parser.add_argument("-v", "--vis", help="visualize (default: False)", default=False, action="store_true")
parser.add_argument("-ct", "--continue-training", help="continue training (default: False)", default=False, action="store_true")
# model
parser.add_argument("-sc", "--scratch", help="train model without pre-trained weights (default: False)", default=False, action="store_true")
parser.add_argument("-m", "--models", help="transfer learning model (default: all)", default="all", nargs='+')
parser.add_argument("-is", "--input_size", help="input size (default: 224)", default=224, type=int)
# train settings
parser.add_argument("-lr", "--learning-rate", help="learning rate (default: 1e-4)", default=1e-4, type=float)
parser.add_argument("-bs", "--batch_size", help="batch size (default: 32)", default=32, type=int)
parser.add_argument("-ep", "--epochs", help="number of epochs (default: 100)", default=100, type=int)
parser.add_argument("-s", "--seed", help="seed number (default: 1998)", default=1998, type=int)
# image processing
#parser.add_argument("-d", "--data-path", help="data path (default: ./data)", default="./data", type=str)
# parse args
args = parser.parse_args()

# =================================
# ========= save location =========
# =================================

def exp_path(exp_name=None, file_name=None):
    out_path = "./results"
    exp_path = os.path.join(out_path, exp_name)
    return exp_path if file_name == None else os.path.join(exp_path, file_name)

# =================================
# ======== public params ==========
# =================================
    
# check for data path to specify the labels
if os.path.exists(args.data_path):
    # extract labels from the data path
    labels = os.listdir(args.data_path)
elif os.path.exists(os.path.join(args.splitted_data_path, "train")):
    # extract labels from the splitted data path
    labels = os.listdir(os.path.join(args.splitted_data_path, "train"))
else:
    raise Exception('problem in data paths in "public params", please run the code properly. for more information: https://github.com/0aub/image-classification')
# count the specified labels
num_classes = len(labels)
# list of all transfer learning from keras applications
transfer_learning_models = {
    "Xception": Xception,
    "VGG16": VGG16,
    "VGG19": VGG19,
    "ResNet50": ResNet50,
    "ResNet50V2": ResNet50V2,
    "ResNet101": ResNet101,
    "ResNet101V2": ResNet101V2,
    "ResNet152": ResNet152,
    "ResNet152V2": ResNet152V2,
    "InceptionV3": InceptionV3,
    "InceptionResNetV2": InceptionResNetV2,
    "MobileNet": MobileNet,
    "MobileNetV2": MobileNetV2,
    "DenseNet121": DenseNet121,
    "DenseNet169": DenseNet169,
    "DenseNet201": DenseNet201,
    "EfficientNetB0": EfficientNetB0,
    "EfficientNetB1": EfficientNetB1,
    "EfficientNetB2": EfficientNetB2,
    "EfficientNetB3": EfficientNetB3,
    "EfficientNetB4": EfficientNetB4,
    "EfficientNetB5": EfficientNetB5,
    "EfficientNetB6": EfficientNetB6,
    "EfficientNetB7": EfficientNetB7,
    "EfficientNetV2B0": EfficientNetV2B0,
    "EfficientNetV2B1": EfficientNetV2B1,
    "EfficientNetV2B2": EfficientNetV2B2,
    "EfficientNetV2B3": EfficientNetV2B3,
    "EfficientNetV2S": EfficientNetV2S,
    "EfficientNetV2M": EfficientNetV2M,
    "EfficientNetV2L": EfficientNetV2L,
}

# =================================
# ======== data filtering =========
# =================================

def filter_images(data_path):
    # remove every file not verified as image
    errors = []
    for dir in os.listdir(data_path):
        label_path = os.path.join(data_path, dir)
        for img in tqdm(os.listdir(label_path), desc=f"filtering {dir} dir: "):
            img_path = os.path.join(label_path, img)
            try:
                img = Image.open(img_path)  # open the image file
                img.verify()  # verify that it is, in fact an image
                if not img_path.lower().endswith(('.png', '.jpg', '.jpeg', '.tiff', '.bmp', '.gif')):
                    raise IOError()
            except (IOError, SyntaxError) as e:
                errors.append(img_path)
                os.remove(img_path)
    print('Deleted images: ', errors)

# =================================
# ====== split data folders =======
# =================================

def split_data_folders(data_path, splitted_data_path, train_ratio, seed):
    if train_ratio > 0 and train_ratio < 1:
        val_ratio = 1 - train_ratio
        # split the data into train/val
        splitfolders.ratio(data_path, output=splitted_data_path, ratio=(train_ratio, val_ratio), seed=seed)
        # data locations
        train_path = os.path.join(splitted_data_path, "train")
        val_path = os.path.join(splitted_data_path, "val")
        return (train_path, val_path)
    else:
        raise Exception("wrong training ratio, please run the code properly. for more information: https://github.com/0aub/image-classification")

# =================================
# ======= data generators =========
# =================================

def data_generators(train_path, val_path, input_size, batch_size, seed):
    # create training and validation generators
    # generators are basically like pytorch data loaders
    train_datagen = ImageDataGenerator()
    train_generator = train_datagen.flow_from_directory(
        train_path,
        target_size=(input_size, input_size),
        batch_size=batch_size,
        class_mode="categorical",
        subset="training",
        shuffle=True,
        seed=seed,
    )

    validation_datagen = ImageDataGenerator()
    validation_generator = validation_datagen.flow_from_directory(
        val_path,
        target_size=(input_size, input_size),
        batch_size=batch_size,
        class_mode="categorical",
        subset="training",
        seed=seed,
    )

    return (train_generator, validation_generator)

# =================================
# ======== data extraction ========
# =================================

def data_extraction(generator, batch_size, title):
    x, y = next(generator)
    for i in tqdm(range(generator.n // batch_size), desc=f"Extracting {title} data: "):
        x_value, y_value = next(generator)
        x = np.append(x, x_value, axis=0)
        y = np.append(y, y_value, axis=0)

    return (x, y)

# =================================
# ===== model initialization ======
# =================================

def TL(TL_model, scratch, lr, input_size, num_classes):
    # generate image shape form its size
    input_shape = (input_size, input_size, 3)
    # create base transfer learning model
    if scratch:
        conv_base = TL_model(include_top=False, weights=None, input_shape=input_shape)
    else:
        conv_base = TL_model(include_top=False, weights="imagenet", input_shape=input_shape)
    # create additional layers for the base model
    top_model = conv_base.output
    top_model = Flatten(name="flatten")(top_model)
    top_model = Dense(128, activation="relu")(top_model)
    top_model = Dropout(0.5)(top_model)
    output_layer = Dense(num_classes, activation="softmax")(top_model)
    # merge the base model with the additional layers
    model = Model(inputs=conv_base.input, outputs=output_layer)
    # for categorical classification
    opt = Adam
    loss_fn = "categorical_crossentropy"
    # for binary classification
    if num_classes == 2:
        opt = SGD
        loss_fn = "binary_crossentropy"
    # build the model with the optimizer and loss function
    model.compile(optimizer=opt(learning_rate=lr), loss=loss_fn, metrics=["accuracy"])
    model.build(input_shape)
    return model

# =================================
# ======== model training =========
# =================================

def train(model, train_generator, validation_generator, batch_size, epochs, exp_name):
    # calculate training steps per epoch
    steps_per_epoch = train_generator.n // batch_size
    validation_steps = validation_generator.n // batch_size
    # train the model and save the progress data in history
    history = model.fit(
        train_generator,
        steps_per_epoch=steps_per_epoch,
        epochs=epochs,
        workers=4,
        validation_data=validation_generator,
        validation_steps=validation_steps,
    )
    # save the model
    model.save(exp_path(exp_name, "model"))
    # save history
    with open(exp_path(exp_name, "history"), "wb") as file:
        pickle.dump(history.history, file)
    # return training/validation progress data
    return history

# =================================
# ======= model evaluation ========
# =================================

def y_true_pred(model, x, y):
    preds = model.predict(x)
    y_pred = y_pred = [1 * (x[0] >= 0.5) for x in preds]
    y_true = np.argmax(y, axis=1)
    return (y_true, y_pred)

def performance_measures(y_true, y_pred):
    TP, FP, TN, FN = (0, 0, 0, 0)
    for i in range(len(y_pred)):
        if y_true[i] == y_pred[i] == 1:
            TP += 1
        if y_pred[i] == 1 and y_true[i] != y_pred[i]:
            FP += 1
        if y_true[i] == y_pred[i] == 0:
            TN += 1
        if y_pred[i] == 0 and y_true[i] != y_pred[i]:
            FN += 1
    return (TP, FP, TN, FN)

def multiclass_processing(y_test, y_pred, average="macro"):
    lb = LabelBinarizer()
    lb.fit(y_test)
    y_test = lb.transform(y_test)
    y_pred = lb.transform(y_pred)
    return (y_test, y_pred)

def evaluate(y_true, y_pred, exp_name):
    processed_y_true, processed_y_pred = multiclass_processing(y_true, y_pred)
    # calculate accuracy and loss
    if num_classes > 2:
        eval = "accuracy:       %.3f" % categorical_accuracy(y_true, y_pred)
    else:
        eval = "accuracy:       %.3f" % binary_accuracy(y_true, y_pred, threshold=0.5)
    # calculate Categorical Cross-Entropy
    #eval += "\nloss:          %.3f" % sum([y_true[i] * math.log(y_pred[i]) for i in range(0, len(y_true))])
    # calculate Mean Square Error (MSE)
    eval += "\nMSE:            %.3f" % metrics.mean_squared_error(y_true, y_pred)
    # calculate Root Mean Square Error (RMSE)
    eval += "\nRMSE:           %.3f" % metrics.mean_squared_error(y_true, y_pred, squared=False)
    # calculate Mean Absolute Error (MAE)
    eval += "\nMAE:            %.3f" % metrics.mean_absolute_error(y_true, y_pred)
    # calculate prediction
    eval += "\nPrecision:      %.3f" % metrics.precision_score(y_true, y_pred, average="macro")
    # calculate recall
    eval += "\nRecall:         %.3f" % metrics.recall_score(y_true, y_pred, average="macro")
    # Calculate Cohen’s Kappa
    eval += "\nCohen’s Kappa:  %.3f" % metrics.cohen_kappa_score(y_true, y_pred)
    # Calculate Matthews correlation coefficient (MCC)
    eval += "\nMCC:            %.3f" % metrics.matthews_corrcoef(y_true, y_pred)
    # Calculate Receiver Operating Characteristic (ROC)
    eval += "\nROC Score:      %.3f" % metrics.roc_auc_score(processed_y_true, processed_y_pred, average="macro")
    # calculate True Positive, True Negative, False Positive and False Negative rates
    TP, FP, FP, FN = performance_measures(y_true, y_pred)
    eval += "\nTrue Positive:  " + str(TP)
    eval += "\nTrue Negative:  " + str(FP)
    eval += "\nFalse Positive: " + str(FP)
    eval += "\nFalse Negative: " + str(FN)
    # print/write the results
    print(eval)
    with open(exp_path(exp_name, "eval.txt"), "w+") as f:
        f.write(eval)

# =================================
# ======= show data balance =======
# =================================

def data_balance_plot(data_path, splitted_data_path, exp_name):
    # count all the images grouping by labels
    counters = []
    for label in labels:
        if os.path.exists(data_path):
            counters.append(len(os.listdir(os.path.join(data_path, label))))
        elif (
        os.path.exists(os.path.join(splitted_data_path, "train")) and 
        os.path.exists(os.path.join(splitted_data_path, "val"))
        ):
            counters.append(len(os.listdir(os.path.join(splitted_data_path, "train", label))) + len(os.listdir(os.path.join(splitted_data_path, "val", label))))
        else:
            raise Exception('cannot plot "data balance plot" without the data path. for more information: https://github.com/0aub/image-classification')
    # printing the results
    #print("".join([f"{label}:  {counter}\n" for (label, counter) in zip(labels, counters)]))
    # pie plot
    plt.figure(figsize=(8, 8))
    plt.pie(counters, labels=labels, colors=["#EE6666", "#3388BB", "#9988DD", "#EECC55", "#88BB44", "#FFBBBB"])
    plt.savefig(exp_path(exp_name, "balance"), dpi=300)

# =================================
# ======= accuracy and loss =======
# =================================

def train_val_history_plot(exp_name):
    # load history
    history = pickle.load(open(exp_path(exp_name, "history"), "rb"))
    # style
    colors = cycler("color", ["#EE6666", "#3388BB", "#9988DD", "#EECC55", "#88BB44", "#FFBBBB"])
    plt.rc("axes", facecolor="#E6E6E6", edgecolor="none", axisbelow=True, grid=True, prop_cycle=colors)
    plt.rc("grid", color="w", linestyle="solid")
    plt.rc("xtick", direction="out", color="gray")
    plt.rc("ytick", direction="out", color="gray")
    plt.rc("patch", edgecolor="#E6E6E6")
    plt.rc("lines", linewidth=2)
    # plot
    acc = history["accuracy"]
    val_acc = history["val_accuracy"]
    loss = history["loss"]
    val_loss = history["val_loss"]
    # plot variables
    plt.figure(figsize=(12, 6))
    plt.plot(acc, label="Training Accuracy")
    plt.plot(val_acc, label="Validation Accuracy")
    plt.plot(loss, label="Training Loss")
    plt.plot(val_loss, label="Validation Loss")
    # plot labels
    plt.ylabel("")
    plt.xlabel("epochs")
    plt.legend(loc="upper right")
    plt.savefig(exp_path(exp_name, "history"), dpi=300)

# =================================
# ======= confusion matrix ========
# =================================

def confusion_matrix_plot(y_true, y_pred, exp_name):
    # confusion matrix
    cm = metrics.confusion_matrix(y_true, y_pred)
    # plot
    fig, ax = plt.subplots(figsize=(10, 10))
    sns.heatmap(cm, annot=True, xticklabels=labels, yticklabels=labels)
    plt.ylabel("Actual")
    plt.xlabel("Predicted")
    plt.savefig(exp_path(exp_name, "cm"), dpi=300)

# ===========================================
# ==== receiver operating characteristic ====
# ===========================================

def roc_plot(y_true, y_pred, labels, exp_name):
    # transform y_true and y_pred
    processed_y_true, processed_y_pred = multiclass_processing(y_true, y_pred)
    # fpr -> False Positive Rate, tpr -> True Positive Rate
    fig, c_ax = plt.subplots(1, 1, figsize=(12, 8))
    for i, label in enumerate(labels):
        fpr, tpr, thresholds = metrics.roc_curve(processed_y_true[:, i], processed_y_pred[:, i])
        c_ax.plot(fpr, tpr, label="%s (AUC:%0.2f)" % (label, metrics.auc(fpr, tpr)))
    plt.title("ROC Curve")
    c_ax.set_xlabel("False Positive Rate")
    c_ax.set_ylabel("True Positive Rate")
    plt.legend(loc="best")
    plt.savefig(exp_path(exp_name, "roc"), dpi=300)

# ==================================
# ===== precision recall curve =====
# ==================================

def pcr_plot(y_true, y_pred, labels, exp_name):
    # transform y_true and y_pred
    processed_y_true, processed_y_pred = multiclass_processing(y_true, y_pred)
    # Plot Precision Recall Curve (PRC)
    fig, _ = plt.subplots(1, 1, figsize=(12, 8))
    precision = dict()
    recall = dict()
    for i, label in enumerate(labels):
        precision[i], recall[i], _ = metrics.precision_recall_curve(processed_y_true[:, i], processed_y_pred[:, i])
        plt.plot(recall[i], precision[i], label=label)
    plt.ylabel("precision")
    plt.xlabel("recall")
    plt.legend(loc="best")
    plt.title("Precision Recall Curve (PRC)")
    plt.savefig(exp_path(exp_name, "pcr"), dpi=300)
    
# =================================
# ===== evaluation collecting =====
# =================================

def collect_evaluations(path):
    # excel columns
    header = ["model", "accuracy", "MSE", "RMSE", "MAE", "Precision", "Recall", "Cohen's Kappa", "MCC", "ROC Score", "True Positive", "True Negative", "False Positive", "False Negative"]
    df = pd.DataFrame(columns=header)
    for model_name in [dirname for dirname in os.listdir(path) if os.path.isdir(os.path.join(path, dirname))]:
        with open(os.path.join(path, model_name, 'eval.txt')) as f:
            row = [line.split(':')[-1].strip() for line in f.readlines()]
            row.insert(0, model_name)
            df = df.append(row)
    df.to_excel(os.path.join(path, "results.xlsx"), index=False)

# =================================
# ========= main function =========
# =================================

def main(args):
    # check if the user splitted the data by himself
    if (os.path.exists(os.path.join(args.splitted_data_path, "train")) and os.path.exists(os.path.join(args.splitted_data_path, "val"))):
        train_path = os.path.join(args.splitted_data_path, "train")
        val_path = os.path.join(args.splitted_data_path, "val")
    # check if the user specified data-path
    elif os.path.exists(args.data_path):
        # filter and verify the images of the data directory
        print("[INFO]  filter the images...\n")
        filter_images(args.data_path)
        # split data folders
        print("\n[INFO]  split to train/val...\n")
        (train_path, val_path) = split_data_folders(args.data_path, args.splitted_data_path, args.training_ratio, args.seed)
    else:
        raise Exception("please run the code properly. for more information: https://github.com/0aub/image-classification")
    
    if args.train or args.eval or args.vis:
        # create data generators
        print("\n[INFO]  create generators...\n")
        (train_generator, validation_generator) = data_generators(train_path, val_path, args.input_size, args.batch_size, args.seed)
        
        # main loop
        models = list(transfer_learning_models.keys()) if 'all' in args.models else args.models
        for model_name in models:
            # create results directory with the current model sub dir if it does not exists
            exp_name = os.path.join(args.exp_name, model_name)
            if not os.path.exists(exp_path(exp_name)):
                os.makedirs(exp_path(exp_name))
            # to continue training 
            elif args.continue_training and os.path.exists(os.path.join(exp_path(exp_name), 'model')):
                    continue

            # model training
            if args.train:
                print(f"\n[INFO]  {model_name} training...\n")
                # get transfer learning object from its name
                TL_model = transfer_learning_models[model_name]
                # initialize the model
                model = TL(TL_model, args.scratch, args.learning_rate, args.input_size, num_classes)
                # train the model
                history = train(
                    model,
                    train_generator,
                    validation_generator,
                    args.batch_size,
                    args.epochs,
                    exp_name,
                )

            # important variables for model evaluation and progress visualization
            if args.eval or args.vis:
                # extract x_val and y_val from validation generator
                print("\n[INFO]  extracting validation values...\n")
                (x_val, y_val) = data_extraction(validation_generator, args.batch_size, "validation")
                # load saved model
                model = load_model(exp_path(exp_name, "model"))
                # get y_true and y_pred
                (y_true, y_pred) = y_true_pred(model, x_val, y_val)

                # model evaluation
                if args.eval:
                    # evaluate the model
                    print(f"\n[INFO]  {model_name} evaluation...\n")
                    evaluate(y_true, y_pred, exp_name)

                # model visualization
                if args.vis:
                    print('\n[INFO]  saving data labels balance...')
                    data_balance_plot(args.data_path, args.splitted_data_path, exp_name)
                    print(f'[INFO]  saving {model_name} training validation progress...')
                    train_val_history_plot(exp_name)
                    print('[INFO]  saving predictions confusion matrix...')
                    confusion_matrix_plot(y_true, y_pred, exp_name)
                    print('[INFO]  saving receiver operating characteristic curve...')
                    roc_plot(y_true, y_pred, labels, exp_name)
                    print('[INFO]  saving precision recall curve...')
                    pcr_plot(y_true, y_pred, labels, exp_name)
                    
        if args.eval:
            collect_evaluations(exp_path(args.exp_name))
            print('\n[INFO]  DONE')

if __name__ == "__main__":
    main(args)
