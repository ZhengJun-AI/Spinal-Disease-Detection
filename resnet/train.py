import os
import torch
import torch.optim as optim
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.utils.data import Dataset
from torchvision import datasets, models, transforms
from torchvision.datasets import ImageFolder
from PIL import Image
import time
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from tqdm import tqdm
import argparse
import logging
from dataloader import creat_loader


parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter,
                                 description='Train resnet-50 model for own data')
parser.add_argument('-n', '--numclass', dest='nc', type=int, default=7,
                    help='Number of categories')
parser.add_argument('-train', '--train_data_dir', dest='train', type=str, default='./images/train/c1/',
                    help='Training data path')
parser.add_argument('-test', '--test_data_dir', dest='test', type=str, default='./images/train/c2/',
                    help='Testing data path')
parser.add_argument('-name', '--model_name', dest='name', type=str, default='mymodel',
                    help='Name of your model')
parser.add_argument('-e', '--epoch', dest='epoch', type=int, default=100,
                    help='Number of epoch to train')
parser.add_argument('-b', '--batch_size', dest='bs', type=int, default=512,
                    help='Batch size of the dataloader')
parser.add_argument('-l', '--learning_rate', dest='lr', type=float, default=0.001,
                    help='Inital learning rate')
parser.add_argument('-r', '--resize', dest='resize', type=int, default=64,
                    help='Images resize to train')
par = parser.parse_args()

logging.basicConfig(level=logging.INFO, format='%(asctime)s : %(message)s',
                    datefmt='%Y-%m-%d %H:%M:%S')

NUM_EPOCH = par.epoch
BATCH_SIZE = par.bs
LEARNING_RATE = par.lr
NUMCLASS = par.nc
RESIZE = par.resize
train_data_dir = par.train
test_data_dir = par.test
modelname = par.name


# train process
def train_and_valid(epochs, model, device, criterion, optimizer, scheduler, train_img_data, test_img_data, train_data_loader, test_data_loader, modelname):
    history = []
    best_acc = 0.0
    best_epoch = 0
 
    for epoch in range(epochs):
        epoch_start = time.time()
        print("Epoch: {}/{}".format(epoch+1, epochs))
        
        model.train()
 
        train_loss = 0.0
        train_acc = 0.0
        valid_loss = 0.0
        valid_acc = 0.0
 
        for i, (inputs, labels) in enumerate(tqdm(train_data_loader)):
            inputs = inputs.to(device)
            labels = labels.to(device)
             
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            
            train_loss += loss.item() * inputs.size(0)
 
            ret, predictions = torch.max(outputs.data, 1)
            correct_counts = predictions.eq(labels.data.view_as(predictions))
 
            acc = torch.mean(correct_counts.type(torch.FloatTensor))
 
            train_acc += acc.item() * inputs.size(0)
 
        with torch.no_grad():
            model.eval()
 
            for j, (inputs, labels) in enumerate(test_data_loader):
                inputs = inputs.to(device)
                labels = labels.to(device)
 
                outputs = model(inputs)
                loss = criterion(outputs, labels)
                valid_loss += loss.item() * inputs.size(0)
 
                ret, predictions = torch.max(outputs.data, 1)
                correct_counts = predictions.eq(labels.data.view_as(predictions))
 
                acc = torch.mean(correct_counts.type(torch.FloatTensor))
 
                valid_acc += acc.item() * inputs.size(0)
        
        train_data_size = len(train_img_data)
        valid_data_size = len(test_img_data)

        avg_train_loss = train_loss/train_data_size
        avg_train_acc = train_acc/train_data_size
 
        avg_valid_loss = valid_loss/valid_data_size
        avg_valid_acc = valid_acc/valid_data_size
 
        history.append([avg_train_loss, avg_valid_loss, avg_train_acc, avg_valid_acc])
 
        if best_acc < avg_valid_acc:
            best_acc = avg_valid_acc
            best_epoch = epoch + 1
            torch.save(model, 'models/c1_best_model-' + modelname + '.pt') # save the best model
 
        epoch_end = time.time()
 
        print("Epoch: {:03d}, Training: Loss: {:.4f}, Accuracy: {:.4f}%, \n\t\tValidation: Loss: {:.4f}, Accuracy: {:.4f}%, Time: {:.4f}s".format(
            epoch+1, avg_train_loss, avg_train_acc*100, avg_valid_loss, avg_valid_acc*100, epoch_end-epoch_start
        ))
        print("Best Accuracy for validation : {:.4f} at epoch {:03d}".format(best_acc, best_epoch))
        
        # torch.save(model, 'models/'+'c1'+'_model_'+str(epoch+1)+'.pkl') #seve each epoch
        scheduler.step() # lr decline
        
    return model, history


# paint result
def showresult(history):
    history = np.array(history, modelname)

    # loss
    plt.plot(history[:, 0:2])
    plt.legend(['Tr Loss', 'Val Loss'])
    plt.xlabel('Epoch Number')
    plt.ylabel('Loss')
    # plt.ylim(0, 1)
    plt.savefig('c1'+'_loss_curve-' + modelname + '.png')
    plt.show()

    # accuracy
    plt.plot(history[:, 2:4])
    plt.legend(['Tr Accuracy', 'Val Accuracy'])
    plt.xlabel('Epoch Number')
    plt.ylabel('Accuracy')
    plt.ylim(0, 1)
    plt.savefig('c1'+'_accuracy_curve-' + modelname + '.png')
    plt.show()


    
if __name__ == '__main__':
    # if use gpu
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    
    # creaet dataloader
    train_img_data, train_data_loader,test_img_data, test_data_loader = creat_loader(train_data_dir, test_data_dir, BATCH_SIZE, RESIZE)
    
    
    # use pretrained-resnet50, and modify the last layer
    resnet50 = models.resnet50(pretrained=True)
    for param in resnet50.parameters():
        param.requires_grad = False

    fc_inputs = resnet50.fc.in_features
    resnet50.fc = nn.Sequential(
        nn.Linear(fc_inputs, 256),
        nn.ReLU(),
        nn.Dropout(0.4),
        nn.Linear(256, NUMCLASS),
    )

    model = resnet50.to(device)

    # define criterion and optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=5, gamma=0.9) # 学习率下降
    
    # train
    trained_model, history = train_and_valid(epochs=NUM_EPOCH, model=model, device=device, criterion=criterion, optimizer=optimizer, scheduler=scheduler, train_img_data=train_img_data, test_img_data=test_img_data, train_data_loader=train_data_loader, test_data_loader=test_data_loader, modelname=modelname)
    
    # show result
    showresult(history, modelname)
    
    logging.info('Process Successfully!')