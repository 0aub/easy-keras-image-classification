# =================================
# ======= import libraries ========
# =================================

import torch
from torch import nn
import torch.optim as optim
from torchvision import datasets, models, transforms

from sklearn.preprocessing import LabelBinarizer
from sklearn import metrics

import numpy as np
import matplotlib.pyplot as plt
from matplotlib import cycler
import seaborn as sns

from PIL import Image
import time
import os
import copy
from tqdm import tqdm
import splitfolders
import argparse
import pickle

# =================================
# ========== arguments ============
# =================================

parser = argparse.ArgumentParser()
# experiment name
parser.add_argument("-en", "--exp-name", help="experiment name (default: mobile-net-v2-exp)", default="mobile-net-v2-exp", type=str)
# data paths
parser.add_argument("-d", "--data-path", help="data path (default: ./data)", default="./data", type=str)
parser.add_argument("-sd", "--splitted-data-path", help="splitted data path (default: ./splitted_data)", default="./splitted_data", type=str)
# run settings
parser.add_argument("-t", "--train", help="train (default: False)", default=False, action="store_true")
parser.add_argument("-e", "--eval", help="eval (default: False)", default=False, action="store_true")
parser.add_argument("-v", "--vis", help="visualize (default: False)", default=False, action="store_true")
# train settings
parser.add_argument("-m", "--model", help="transfer learning model (default: MobileNetV2)", default="MobileNetV2", type=str)
parser.add_argument("-lr", "--learning-rate", help="learning rate (default: 1e-4)", default=1e-4, type=float)
parser.add_argument("-bs", "--batch_size", help="batch size (default: 32)", default=32, type=int)
parser.add_argument("-e", "--epochs", help="number of epochs (default: 100)", default=100, type=int)
parser.add_argument("-is", "--input_size", help="input size (default: 224)", default=224, type=int)
# parse args
args = parser.parse_args()

# =================================
# ========= save location =========
# =================================

def exp_save(exp_name=None, file_name=None):
  out_path = './results' 
  exp_path = os.path.join(out_path, exp_name)
  return exp_path if file_name == None else os.path.join(exp_path, file_name)

# =================================
# ======== public params ==========
# =================================

# create results directory if it does not exists
if not os.path.exists(exp_save(args.exp_name)):
    os.makedirs(exp_save(args.exp_name))
    
labels = None

# =================================
# ========= encoder model =========
# =================================

class Encoder(nn.Module):
    """ 
    Takes a image and returns an encoded reprsentation of that image. 
    """

    def __init__(self, enc_img_size):
        """
        Initializes the Encoder Layer 
        Params: 
            enc_img_size: the size of the encoded image
                         after passing thourgh the CNN Model
        Returns: 
            output of shape (batch_size, encode_out_num_channels, enc_img_size*enc_img_size)
        """
        super(Encoder, self).__init__()
        self.enc_img_size = enc_img_size
        arch = 'resnet50'

        model_file = '%s_places365.pth.tar' % arch
        if not os.access(model_file, os.W_OK):
            weight_url = 'http://places2.csail.mit.edu/models_places365/' + model_file
            os.system('wget ' + weight_url)

        model = models.__dict__[arch](num_classes=365)
        checkpoint = torch.load(model_file, map_location=lambda storage, loc: storage)
        state_dict = {str.replace(k,'module.',''): v for k,v in checkpoint['state_dict'].items()}
        model.load_state_dict(state_dict)
        self.num_channels = 2048

        # remove the last pool and fc layer 
        modules = list(model.children())[:-2]
        
        # remake the network 
        self.resnet = nn.Sequential(*modules)
        self.pool = nn.AdaptiveAvgPool2d(enc_img_size)


    def forward(self, images):
        """
        Does a forwad pass over the encoder model
        Params: 
            images = images from the dataset
        """
        out = self.resnet(images)
        out = self.pool(out)
        encoder_out = out.view(-1, self.num_channels, self.enc_img_size*self.enc_img_size)
        # (batch_size, encode_out_num_channels, enc_img_size*enc_img_size)
        return encoder_out

# =================================
# ======== attention model ========
# =================================

class Attention(nn.Module):
    """
    Attention Mask over the encoded image produced by the CNN. 
    It takes the input of the LSTM Cell state to update its values & 
    encoded image.
    We will use a hidden Layer to transform the output of
    the LSTM to the dimension the encoded image.
    After that softmax will be applied on that vector. 
    After that element wise product of the image and the attention vector
    will be taken. This vector will be passed to the LSTM again. 
    """

    def __init__(self, num_channels, encoder_dim, decoder_dim):
        """
        Initialize the Attention Layer. 
        Args:
            num_channels -> number of channels in the output of the CNN model
            encoder_dim -> the height or width dimension of the output of the CNN Model (num_channels, H, W). 
            decoder_dim -> the dimension of the output of the decoder.
        """
        super(Attention, self).__init__()
        self.encoder_dim = encoder_dim
        self.num_channels = num_channels
        # linear layer to transform decoder's output to attention dimension
        self.decoder_att = nn.Linear(decoder_dim, encoder_dim*encoder_dim)
        # softmax layer over the output of the above linear layer
        self.softmax = nn.Softmax(dim=1) 
        # avg. pool the output after applying the softmax over it
        self.avg_pool = nn.AdaptiveAvgPool2d(1)

    def forward(self, encoder_out, decoder_hidden):
        """
        Does a forward pass.
        Args: 
            encoder_out -> the output of the encoder (batch_size, encode_out_num_channels, encoder_dim*encoder_dim)
            decoder_hidden -> the output of the decoder (batch_size, decoder_dim)
        Returns: 
            attention_weighted_encoding (batch_size, encode_out_num_channels)
        """

        # (batch_size, encoder_dim*encoder_dim)
        decoder2att = self.decoder_att(decoder_hidden)
        # (batch_size, encoder_dim*encoder_dim)
        attn = self.softmax(decoder2att)

        # calculate the attention over the image
        # (batch_size, encode_out_num_channels,  encoder_dim*encoder_dim)
        attention_weighted_encoding = encoder_out*(attn.unsqueeze(1))
        # Reshape to (batch_size, encode_out_num_channels, encoder_dim, encoder_dim)
        attention_weighted_encoding = attention_weighted_encoding.view(-1, self.num_channels, self.encoder_dim, self.encoder_dim)
        # apply Adaptive Avg. Pool 
        pooled = self.avg_pool(attention_weighted_encoding) # (batch_size, encode_out_num_channels, 1, 1)
        attention_weighted_encoding = pooled.view(-1, self.num_channels) # (batch_size, encode_out_num_channels)
        return attention_weighted_encoding

# =================================
# ========= decoder model =========
# =================================

class Decoder(nn.Module):
    """
    This will recieve the input from the Attention Layer & encoder,
    that will be passed on to the LSTM cell to do it's work.
    """

    def __init__(self, num_channels, encoder_dim, decoder_dim1, decoder_dim2, decoder_dim3, num_classes, num_lstm_cell):
        """
        Args: 
        num_channels -> num_channels in the output of the CNN model  
        encoder_dim ->  the height or width dimension of the output of the CNN Model (num_channels, H, W).
        decoder_dim1 -> the dim. of the decoder1
        decoder_dim2 -> the dim. of the decoder2 
        decoder_dim3 -> the dim. of the decoder3 
        num_lstm_cell -> the recurrence number
        num_classes -> the number of classes
        """
        super(Decoder, self).__init__()

        self.num_classes = num_classes
        self.num_lstm_cell = num_lstm_cell

        self.attention = Attention(num_channels, encoder_dim, decoder_dim1)
        self.decode_step1 = nn.LSTMCell(num_channels, decoder_dim1, bias=True)
        self.decode_step2 = nn.LSTMCell(decoder_dim1, decoder_dim2, bias=True)
        self.decode_step3 = nn.LSTMCell(decoder_dim2, decoder_dim3, bias=True)
        self.last_linear = nn.Linear(decoder_dim3, num_classes)
        self.softmax = nn.Softmax(dim = 1)

        self.init_h1 = nn.Linear(encoder_dim*encoder_dim, decoder_dim1)  # linear layer to find initial hidden state of LSTMCell-1
        self.init_c1 = nn.Linear(encoder_dim*encoder_dim, decoder_dim1) 
        self.init_h2 = nn.Linear(encoder_dim*encoder_dim, decoder_dim2)  # linear layer to find initial hidden state of LSTMCell-2
        self.init_c2 = nn.Linear(encoder_dim*encoder_dim, decoder_dim2) 
        self.init_h3 = nn.Linear(encoder_dim*encoder_dim, decoder_dim3)  # linear layer to find initial hidden state of LSTMCell-3
        self.init_c3 = nn.Linear(encoder_dim*encoder_dim, decoder_dim3) 
        

    def init_hidden_state(self, encoder_out):
        """
        Initializes the param of h, c for 3 stacked LSTM 
        Args: 
            encoder_out -> the output of the encoder (batch_size, encode_out_num_channels, encoder_dim*encoder_dim)
        """
        mean_encoder_out = encoder_out.mean(dim=1)
        h1 = self.init_h1(mean_encoder_out)  # (batch_size, decoder_dim)
        c1 = self.init_c1(mean_encoder_out)

        h2 = self.init_h2(mean_encoder_out)  # (batch_size, decoder_dim)
        c2 = self.init_c2(mean_encoder_out)

        h3 = self.init_h3(mean_encoder_out)  # (batch_size, decoder_dim)
        c3 = self.init_c3(mean_encoder_out)
        return h1, c1, h2, c2, h3, c3

    def forward(self, encoder_out):
        """
        Does a forward pass.
        Args:
            encoder_out -> the output of the encoder (batch_size, encode_out_num_channels, encoder_dim*encoder_dim)
        """
        
        batch_size = encoder_out.size(0)
        h1, c1, h2, c2, h3, c3  = self.init_hidden_state(encoder_out)

        # stores the output of the LSTM Cells (We are having num_lstm_cell number of cells)
        y_complete = torch.zeros(size = (self.num_lstm_cell, batch_size, self.num_classes))

        for i in range(self.num_lstm_cell):
            attention_weighted_encoding = self.attention(encoder_out, h1) # (batch_size, encode_out_num_channels)
            h1, c1 = self.decode_step1(attention_weighted_encoding, (h1, c1)) # (batch_size, decoder_dim1), # (batch_size, decoder_dim1)
            h2, c2 = self.decode_step2(h1, (h2, c2)) # (batch_size, decoder_dim2), # (batch_size, decoder_dim2)  
            h3, c3 = self.decode_step3(h2, (h3, c3)) # (batch_size, decoder_dim3), # (batch_size, decoder_dim3)
            out = self.last_linear(h3) # (batch_size, num_classes)
            y_t = self.softmax(out) # (batch_size, num_classes)
            y_complete[i] = y_t

        return y_complete.sum(dim =0 )

# =================================
# ======== data filtering =========
# =================================

def filter_images(data_path):
    # remove every file not verified as image
    for dir in os.listdir(data_path):
      label_path = os.path.join(data_path, dir)
      for img in tqdm(os.listdir(label_path), desc=f'filtering {dir} dir: '):
          img_path = os.path.join(label_path, img)
          try:
              img = Image.open(img_path)  # open the image file
              img.verify()  # verify that it is, in fact an image
          except (IOError, SyntaxError) as e:
              print(img_path)
              os.remove(img_path)

# =================================
# ====== split data folders =======
# =================================

def split_data_folders(data_path, splitted_data_path, seed):
    # split the data into train/val
    splitfolders.ratio(data_path, output=splitted_data_path, ratio=(0.8, 0.2), seed=seed) 
    # data locations 
    train_path = os.path.join(splitted_data_path, "train")
    val_path = os.path.join(splitted_data_path, "val")

    return (train_path, val_path)

# =================================
# ==== data objects extraction ====
# =================================

def data_objects_extraction(data_path, batch_size):
   # data transforms
    data_transforms = {
        'train': transforms.Compose([
            transforms.RandomResizedCrop((256,256)),
            transforms.RandomHorizontalFlip(.2),
            transforms.RandomVerticalFlip(.3),
            transforms.RandomRotation(5),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ]),
        'val': transforms.Compose([
            transforms.Resize((256,256)),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ]),
    }
    # images dataset object
    image_datasets = {x: datasets.ImageFolder(os.path.join(data_path, x), data_transforms[x]) for x in ['train', 'val']}
    # data loaders -> like data generators in tensorflow
    dataloaders = {x: torch.utils.data.DataLoader(image_datasets[x], batch_size=32, shuffle=True) for x in ["train", "val"]}
    # get the length of training dataset and validation dataset
    dataset_sizes = {x: len(image_datasets[x]) for x in ['train', 'val']}
    # extract classes names
    class_names = image_datasets['train'].classes
    # return the used object in training function
    return (dataloaders, dataset_sizes, class_names)

# =================================
# ===== model initialization ======
# =================================

def get_model():
    # create torch device
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    # add encoder model to torch device
    encoder = Encoder(8).to(device)
    # add decoder model to torch device
    decoder = Decoder(2048, 8, 32, 64, 128, 31, 30).to(device)
    # add loss function to torch device
    criterion = nn.CrossEntropyLoss().to(device)
    # optimizer parameters
    plist = [
             {'params': encoder.parameters(), 'lr': 1e-5, "weight_decay": 1e-4},
             {'params': decoder.parameters(), 'lr': 1e-3, "weight_decay": 1e-4}
            ]
    # adam optimizer
    optimizer = optim.Adam(plist)
    # return the used object in training function
    return (encoder, decoder, criterion, optimizer)

# =================================
# ==== training history values ====
# =================================

def history_values(history):

    # extract training and validation accuracy
    train_accuracy= history["accuracy"][0]
    val_accuracy= history["accuracy"][1]
    # extract training and validation loss
    train_loss= history["loss"][0]
    val_loss= history["loss"][1]
    # convert all accuracy values to np array
    train_accuracy = [value.cpu().numpy() for value in train_accuracy]
    val_accuracy = [value.cpu().numpy() for value in val_accuracy]
    # return extracted training history values
    return (train_accuracy, val_accuracy, train_loss, val_loss)

# =================================
# ======== model training =========
# =================================

def train(encoder, decoder, criterion, optimizer, dataloaders, dataset_sizes, class_names, epochs, exp_name):
    # create torch device
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    # start timer
    since = time.time()
    # deep copy of the model to start training
    best_decoder = copy.deepcopy(decoder.state_dict())
    best_accuracy = 0.0
    # empty lists for training history
    val_accuracy_history = []
    val_loss_history = []
    train_accuracy_history = []
    train_loss_history = []
    # main training loop
    for epoch in range(epochs):
        # print current epoch
        print('Epoch {}/{}'.format(epoch, epochs - 1))
        print('-' * 10)

        # Each epoch has a training and validation phase
        for phase in ['train', 'val']:
            if phase == 'train':
                # Set model to training mode
                encoder.train() 
                decoder.train()
            else:
                # Set model to evaluate mode
                encoder.eval()   
                decoder.eval()

            # restart current loss and correct predictions
            running_loss = 0.0
            running_corrects = 0

            # Iterate over data.
            for inputs, labels in tqdm(dataloaders[phase], total=len(dataloaders[phase])):
                # add features and labels to torch device
                inputs = inputs.to(device)
                labels = labels.to(device)
                # zero the parameter gradients
                optimizer.zero_grad()

                # forward
                # track history if only in train
                with torch.set_grad_enabled(phase == 'train'):
                    # predictions
                    inputs = encoder(inputs)
                    outputs = decoder(inputs).to(device)
                    _, preds = torch.max(outputs, 1)
                    # get losses
                    loss = criterion(outputs, labels)
                    # backward + optimize only if in training phase
                    if phase == 'train':
                        loss.backward()
                        optimizer.step()

                # accuracy and loss calculations
                running_loss += loss.item() * inputs.size(0)
                running_corrects += torch.sum(preds == labels.data)

            # calculate average epoch accuracy and loss
            loss = running_loss / dataset_sizes[phase]
            accuracy = running_corrects.double() / dataset_sizes[phase]
            print('{} Loss: {:.4f} Acc: {:.4f}'.format(phase, loss, accuracy))

            # deep copy the model
            if phase == 'val' and accuracy > best_accuracy:
                best_accuracy = accuracy
                best_decoder = copy.deepcopy(decoder.state_dict())
                best_encoder = copy.deepcopy(encoder.state_dict())
            # store validation accuracy and loss
            if phase == "val":
                val_accuracy_history.append(accuracy)
                val_loss_history.append(loss)
            # store training accuracy and loss
            if phase == "train":
                train_accuracy_history.append(accuracy)
                train_loss_history.append(loss)

        # save checkpoint
        torch.save({
            "model": decoder.state_dict(),
            "best_model": best_decoder,
            "encoder": encoder.state_dict(),
            "best_encoder": best_encoder,
            "optimizer": optimizer.state_dict(),
            "epoch": epoch,
            "history": {
                "loss": [train_loss_history, val_loss_history], 
                "accuracy": [train_accuracy_history, val_accuracy_history]
                }
             },
            "model.pth"
        )

    # training summary
    time_elapsed = time.time() - since
    print('Training complete in {:.0f}m {:.0f}s'.format(time_elapsed // 60, time_elapsed % 60))
    print('Best val accuracy: {:4f}'.format(best_accuracy))
    # load best model weights
    encoder.load_state_dict(best_encoder)
    decoder.load_state_dict(best_decoder)
    # save models
    torch.save(encoder.state_dict(), exp_save(exp_name, 'best_encoder.pth'))
    torch.save(decoder.state_dict(), exp_save(exp_name, 'best_decoder.pth'))
    # save model history
    history = {"loss": [train_loss_history, val_loss_history], "accuracy": [train_accuracy_history, val_accuracy_history]}
    with open(exp_save(exp_name, 'history'), 'wb') as file:
        pickle.dump(history, file)
    # return decoder model, encoder model, and training progress history
    #return (encoder, decoder, history)
    print('DONE')

# =================================
# ======= model evaluation ========
# =================================

def performance_measures(y_true, y_pred):
    TP, FP, TN, FN = (0, 0, 0, 0)
    for i in range(len(y_pred)): 
        if y_true[i]==y_pred[i]==1:
           TP += 1
        if y_pred[i]==1 and y_true[i]!=y_pred[i]:
           FP += 1
        if y_true[i]==y_pred[i]==0:
           TN += 1
        if y_pred[i]==0 and y_true[i]!=y_pred[i]:
           FN += 1
    return (TP, FP, TN, FN)

def multiclass_processing(y_test, y_pred, average="macro"):
    lb = LabelBinarizer()
    lb.fit(y_test)
    y_test = lb.transform(y_test)
    y_pred = lb.transform(y_pred)
    return (y_test, y_pred)

def measures(y_true, y_pred, exp_name):
    # start variable
    eval = '\n\n'
    # calculate accuracy and loss
    eval += 'Accuracy:        %.3f' % metrics.accuracy_score(y_true, y_pred)
    # calculate prediction
    eval += '\nPrecision:     %.3f' % metrics.precision_score(y_true, y_pred, average='macro')
    # calculate recall
    eval += '\nRecall:        %.3f' % metrics.recall_score(y_true, y_pred, average='macro')
    # Calculate Cohen’s Kappa
    eval += '\nCohen’s Kappa: %.3f' % metrics.cohen_kappa_score(y_true, y_pred)
    # Calculate Matthews correlation coefficient (MCC)
    eval += '\nMCC:           %.3f' % metrics.matthews_corrcoef(y_true, y_pred)
    # Calculate Receiver Operating Characteristic (ROC)
    processed_y_true, processed_y_pred = multiclass_processing(y_true, y_pred)
    eval += '\nROC Score :    %.3f' % metrics.roc_auc_score(processed_y_true, processed_y_pred, average='macro')
    # calculate True Positive, True Negative, False Positive and False Negative rates
    TP, FP, FP, FN = performance_measures(y_true, y_pred)
    eval += '\nTrue Positive: ' + str(TP) + ' False Positive: ' + str(FP)
    eval += '\nTrue Negative: ' + str(FP) + ' False Negative: ' + str(FN)
    # print/write the results
    print(eval)
    with open(exp_save(exp_name, 'eval.txt'), 'w+') as f:
        f.write(eval)

def classes_accuracy(class_names, class_correct, class_total, exp_name):
    # start variable
    eval = ''
    # calculate the accuracy for each class
    for i in range(len(class_names)):
        eval += '\nAccuracy of %5s : %2d %%' % (class_names[i], (100 * class_correct[i] / class_total[i] if class_total[i] > 0 else 0.0 ))
    # print/write the results
    print(eval)
    with open(exp_save(exp_name, 'eval.txt'), 'w+') as f:
        f.write(eval)

def predictions(encoder, decoder, dataloader, class_names, batch_size, exp_name):
    # create torch device
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    # load best weights of encoder and decoder models
    encoder.load_state_dict(torch.load(exp_save(exp_name, 'best_encoder.pth')))
    decoder.load_state_dict(torch.load(exp_save(exp_name, 'best_decoder.pth')))
    # start variables
    corrects = 0
    total = 0
    class_correct = list(0. for i in range(len(class_names)))
    class_total = list(0. for i in range(len(class_names)))
    preds = []
    true_labels = []
    images_list = []
    # model evaluation
    with torch.no_grad():
        for data in dataloader:
            inputs, labels = data
            inputs, labels = inputs.to(device), labels.to(device)
            out_encoder_model = encoder(inputs)
            out_decoder_model = decoder(out_encoder_model)
            _, predicted = torch.max(out_decoder_model, 1)
            predicted = predicted.to(device)
            total += labels.size(0)
            corrects += (predicted == labels).sum().item()
            c = (predicted == labels).squeeze()
            for i in range(4):
                preds.append(predicted[i])
                images_list.append(inputs[i])
                label = labels[i]
                true_labels.append(label)
                class_correct[label] += c[i].item()
                class_total[label] += 1
    # return the used variables in the evaluation and visualization
    return (true_labels, preds, class_correct, class_total)

def evaluate(true_labels, preds, class_names, class_correct, class_total, exp_name):
    # calculate/save accuracy and other measures
    classes_accuracy(class_names, class_correct, class_total, exp_name)
    measures(true_labels, preds, exp_name)

# =================================
# ======= show data balance =======
# =================================

def data_balance_plot(data_path, exp_name):
    # extract labels from the data path
    labels = os.listdir(args.data_path)
    # count all the images grouping by labels
    counters = [len(os.listdir(os.path.join(data_path, label))) for label in labels]
    # printing the results
    print(''.join([f'{label}:  {counter}\n' for (label, counter) in zip(labels, counters)]))

    # pie plot
    plt.figure(figsize=(8, 8))
    plt.pie(counters, 
            labels = labels, 
            colors = ['#EE6666', '#3388BB', '#9988DD','#EECC55', '#88BB44', '#FFBBBB'])
    plt.savefig(exp_save(exp_name, 'balance'), dpi=300)
    plt.show()

# =================================
# ======= accuracy and loss =======
# =================================

def train_val_history_plot(exp_name):
    # load history
    history = pickle.load(open(exp_save(exp_name, 'history'), "rb"))
    (train_accuracy, val_accuracy, train_loss, val_loss) = history_values(history)
    # style
    colors = cycler('color',['#EE6666', '#3388BB', '#9988DD','#EECC55', '#88BB44', '#FFBBBB'])
    plt.rc('axes', facecolor='#E6E6E6', edgecolor='none', axisbelow=True, grid=True, prop_cycle=colors)
    plt.rc('grid', color='w', linestyle='solid')
    plt.rc('xtick', direction='out', color='gray')
    plt.rc('ytick', direction='out', color='gray')
    plt.rc('patch', edgecolor='#E6E6E6')
    plt.rc('lines', linewidth=2)

    # plot
    accuracy = train_accuracy
    val_accuracy = val_accuracy
    loss = train_loss
    val_loss = val_loss

    plt.figure(figsize=(12, 6))
    plt.plot(accuracy, label="Training Accuracy")
    plt.plot(val_accuracy, label="Validation Accuracy")
    plt.plot(loss, label="Training Loss")
    plt.plot(val_loss, label="Validation Loss")

    plt.ylabel('')
    plt.xlabel('epochs')
    plt.legend(loc="upper right")
    plt.savefig(exp_save(exp_name, 'history'), dpi=300)
    plt.show()

# =================================
# ======= confusion matrix ========
# =================================

def confusion_matrix_plot(y_true, y_pred, exp_name):
    # confusion matrix
    cm = metrics.confusion_matrix(y_true, y_pred)
    # plot
    fig, ax = plt.subplots(figsize=(10,10))
    sns.heatmap(cm, annot=True, xticklabels=labels, yticklabels=labels)
    plt.ylabel('Actual')
    plt.xlabel('Predicted')
    plt.savefig(exp_save(exp_name, 'cm'), dpi=300)
    plt.show()

# ===========================================
# ==== receiver operating characteristic ====
# ===========================================

def roc_plot(y_true, y_pred, labels, exp_name):
    # transform y_true and y_pred
    processed_y_true, processed_y_pred = multiclass_processing(y_true, y_pred)
    # fpr -> False Positive Rate, tpr -> True Positive Rate
    fig, c_ax = plt.subplots(1,1, figsize = (12, 8))
    for i, label in enumerate(labels):
        fpr, tpr, thresholds = metrics.roc_curve(processed_y_true[:, i], processed_y_pred[:, i])
        c_ax.plot(fpr, tpr, label='%s (AUC:%0.2f)' % (label, metrics.auc(fpr, tpr)))
    plt.title('ROC Curve')
    c_ax.set_xlabel('False Positive Rate')
    c_ax.set_ylabel('True Positive Rate')
    plt.legend(loc="best")
    plt.savefig(exp_save(exp_name, 'roc'), dpi=300)
    plt.show()

# ==================================
# ===== precision recall curve =====
# ==================================

def pcr_plot(y_true, y_pred, labels, exp_name):
    # transform y_true and y_pred
    processed_y_true, processed_y_pred = multiclass_processing(y_true, y_pred)
    # Plot Precision Recall Curve (PRC)
    fig, _ = plt.subplots(1,1, figsize = (12, 8))
    precision = dict()
    recall = dict()
    for i, label in enumerate(labels):
        precision[i], recall[i], _ = metrics.precision_recall_curve(processed_y_true[:, i], processed_y_pred[:, i])
        plt.plot(recall[i], precision[i], label=label)
    plt.ylabel("precision")
    plt.xlabel("recall")
    plt.legend(loc="best")
    plt.title("Precision Recall Curve (PRC)")
    plt.savefig(exp_save(exp_name, 'pcr'), dpi=300)
    plt.show()
    
# =================================
# ========= main function =========
# =================================

def main():
    # check if the user specified data-path
    if args.data_path:
        # filter and verify the images of the data directory
        print('[INFO]  filter the images...')
        filter_images(args.data_path)
        # split data folders
        print('\n[INFO]  split to train/val...')
        (train_path, val_path) = split_data_folders(args.data_path, args.splitted_data_path, seed=args.seed)
    if (
        os.path.exists(os.path.join(args.splitted_data_path, "train")) and 
        os.path.exists(os.path.join(args.splitted_data_path, "val"))
    ):
        # get important data objects from its path
        print('\n[INFO]  extract data objects...')
        global labels
        (dataloaders, dataset_sizes, labels) = data_objects_extraction(args.splitted_data_path, args.batch_size)
    else:
        raise Exception("please run the code properly. for more information: https://github.com/0aub/")

    # initialize encoder, decoder, loss function, and optimizer
    print('\n[INFO]  initialize model...')
    (encoder, decoder, criterion, optimizer) = get_model()

    # model training
    if args.train:
        print('\n[INFO]  model training...')
        train(encoder, decoder, criterion, optimizer, dataloaders, dataset_sizes, labels, args.epochs, args.exp_name)

    # get predictions of validation data for model evaluation and visualization
    if args.eval or args.vis:
        (y_true, y_pred, class_correct, class_total) = predictions(encoder, decoder, dataloaders['val'], labels, args.batch_size, args.exp_name)
        # convert cuda tensors to integers predictions
        y_true = [int(x.cpu()) for x in y_true]
        y_pred = [int(x.cpu()) for x in y_pred]

        # model evaluation -> calculate/save accuracy and other measures
        if args.eval:
            print('\n[INFO]  model evaluation...')
            evaluate(y_true, y_pred, labels, class_correct, class_total, args.exp_name)

        # model visualization
        if args.vis:
            print('\n[INFO]  plotting data labels balance...\n')
            data_balance_plot(args.data_path, args.exp_name)
            print('\n[INFO]  plotting model training validation progress...\n')
            train_val_history_plot(args.exp_name)
            print('\n[INFO]  plotting predictions confusion matrix...\n')
            confusion_matrix_plot(y_true, y_pred, args.exp_name)
            print('\n[INFO]  plotting receiver operating characteristic curve...\n')
            roc_plot(y_true, y_pred, labels, args.exp_name)
            print('\n[INFO]  plotting precision recall curve...\n')
            pcr_plot(y_true, y_pred, labels, args.exp_name)
            
if __name__ == "__main__":
    main(args)
