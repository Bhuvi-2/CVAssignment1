# import packages here
import cv2
import numpy as np
import matplotlib.pyplot as plt
import glob
import random
import time

import torch
import torchvision
import torchvision.transforms as transforms

from torch.autograd import Variable
import torch.nn as nn
import torch.nn.functional as F
import albumentations
import torchvision.models as models
import torch.optim as optim
import time

#--------------------------------------------------
#    Load Training Data and Testing Data
#--------------------------------------------------

import albumentations

def set_random_seed(seed):
   torch.manual_seed(seed)
   torch.cuda.manual_seed_all(seed)
   np.random.seed(seed)
   random.seed(seed)
   torch.backends.cudnn.deterministic = True
set_random_seed(0)

class_names = [name[13:] for name in glob.glob('./data/train/*')]
class_names = dict(zip(range(len(class_names)), class_names))
print("class_names: %s " % class_names)

def load_dataset(path, img_size, num_per_class=-1, batch_size=16, shuffle=False,
       augment=False, is_color=False, zero_centered=False, mirror=False, rotate=False, together=False):
  set_random_seed(0)
  data = []
  labels = []
  channel_num = 3 if is_color else 1

  # read images and resizing
  for id, class_name in class_names.items():
    print("Loading images from class: %s" % id)
    img_path_class = glob.glob(path + class_name + '/*.jpg')
    if num_per_class > 0:
      img_path_class = img_path_class[:num_per_class]
    labels.extend([id]*len(img_path_class))
    for filename in img_path_class:
      if is_color:
        img = cv2.imread(filename)
      else:
        img = cv2.imread(filename, 0)

      # resize the image
      img = cv2.resize(img, img_size, cv2.INTER_LINEAR)

      if is_color:
        img = np.transpose(img, [2, 0, 1])

      # norm pixel values to [-1, 1]
      data.append(img.astype(np.float64)/255*2-1)

  # Data Augmentation

  if augment:
    aug_mirror = albumentations.Compose(
      [
        albumentations.HorizontalFlip(p=1),
      ],
    )

    aug_rotate = albumentations.Compose(
      [
        albumentations.Rotate(p=1, border_mode=0, limit=30),
      ],
    )


    dataset_length = len(data)

    if mirror and not rotate:
      for i in range(dataset_length):

        img_temp = aug_mirror(image=data[i])['image']

        data.append(img_temp)
        labels.append(labels[i])


    if rotate and not mirror:
      for i in range(dataset_length):

        img_temp = aug_rotate(image=data[i])['image']

        data.append(img_temp)
        labels.append(labels[i])


    if  mirror and rotate:
      for i in range(dataset_length):

        img_temp = aug_mirror(image=data[i])['image']

        data.append(img_temp)
        labels.append(labels[i])

      for i in range(dataset_length):

        img_temp = aug_rotate(image=data[i])['image']

        data.append(img_temp)
        labels.append(labels[i])


    if together:
      for i in range(dataset_length):

        img_temp = aug_mirror(image=data[i])['image']
        img_temp = aug_rotate(image=img_temp)['image']

        data.append(img_temp)
        labels.append(labels[i])


  # Data Normalization
  if zero_centered:
    if is_color:
      for i,image in enumerate(data):
        data[i][0] += np.mean(image[0])
        data[i][1] += np.mean(image[1])
        data[i][2] += np.mean(image[2])
    else:
      for i,image in enumerate(data):
        data[i] += np.mean(image)
    # pass


  # randomly permute (this step is important for training)
  if shuffle:
    bundle = list(zip(data, labels))
    random.shuffle(bundle)
    data, labels = zip(*bundle)

  # divide data into minibatches of TorchTensors
  if batch_size > 1:
    batch_data = []
    batch_labels = []

    for i in range(int(len(data) / batch_size)):
      minibatch_d = data[i*batch_size: (i+1)*batch_size]
      minibatch_d = np.reshape(minibatch_d, (batch_size, channel_num, img_size[0], img_size[1]))
      batch_data.append(torch.from_numpy(minibatch_d))

      minibatch_l = labels[i*batch_size: (i+1)*batch_size]
      batch_labels.append(torch.LongTensor(minibatch_l))
    data, labels = batch_data, batch_labels

  return zip(batch_data, batch_labels)

# load data into size (64, 64)
img_size = (224, 224)
batch_size = 32 # training sample number per batch

# load training dataset
trainloader_small = list(load_dataset('./data/train/', img_size, batch_size=batch_size, shuffle=True, augment=True, mirror=True, rotate=True, zero_centered=True))
train_num = len(trainloader_small)
print("Finish loading %d minibatches (batch_size=%d) of training samples." % (train_num, batch_size))

# load testing dataset
testloader_small = list(load_dataset('./data/test/', img_size, num_per_class=50, batch_size=batch_size))
test_num = len(testloader_small)
print("Finish loading %d minibatches (batch_size=%d) of testing samples." % (test_num, batch_size))

# show some images
def imshow(img):
    img = img / 2 + 0.5     # unnormalize
    npimg = img.numpy()
    if len(npimg.shape) > 2:
        npimg = np.transpose(img, [1, 2, 0])
    plt.figure
    plt.imshow(npimg, 'gray')
    plt.show()
img, label = trainloader_small[0][0][11][0], trainloader_small[0][1][11]
label = int(np.array(label))
#print(class_names[label])
#imshow(img)

#--------------------------------------------------
#       Define Network Architecture
#--------------------------------------------------
class TNet(nn.Module):
    def __init__(self, num_classes=16):
        super(TNet, self).__init__()
        
        # Load the pre-trained ResNet model
        self.resnet = models.resnet50(pretrained=True)
        
        # Modify the first convolutional layer to accept 1-channel input if necessary
        self.resnet.conv1 = nn.Conv2d(1, 64, kernel_size=7, stride=2, padding=3, bias=False)
        
        # Freeze all layers initially
        for param in self.resnet.parameters():
            param.requires_grad = False
        
        # Unfreeze the last few layers (example: layer4 and fc)
        for param in self.resnet.layer4.parameters():
            param.requires_grad = True
        for param in self.resnet.fc.parameters():
            param.requires_grad = True
        
        # Replace the final fully connected layer to match the number of classes
        self.resnet.fc = nn.Linear(self.resnet.fc.in_features, num_classes)
        
        # Ensure the final layer's parameters are trainable
        for param in self.resnet.fc.parameters():
            param.requires_grad = True

    def forward(self, x):
        return self.resnet(x)
    
#--------------------------------------------------
#       Define Another Network Architecture
#--------------------------------------------------
    
class TNet2(nn.Module):
    def __init__(self, num_classes=16):
        super(TNet2, self).__init__()
        
        # Load the pre-trained DenseNet121 model
        self.densenet = models.densenet121(pretrained=True)
        
        # Modify the first convolutional layer to accept 1-channel input if necessary
        self.densenet.features.conv0 = nn.Conv2d(1, 64, kernel_size=5, stride=2, padding=3, bias=False)
        
        # Freeze all layers initially
        for param in self.densenet.parameters():
            param.requires_grad = False
        
        # Replace the final fully connected layer to match the number of classes
        self.densenet.classifier = nn.Sequential(
            nn.Dropout(0.5),
            nn.Linear(self.densenet.classifier.in_features, num_classes)
        )
        
        # Ensure the final layer's parameters are trainable
        for param in self.densenet.classifier.parameters():
            param.requires_grad = True

    def forward(self, x):
        x = self.densenet(x)
        return x
    
#--------------------------------------------------
#       Define VGG16 Network Architecture
#--------------------------------------------------

class TNet3(nn.Module):
    def __init__(self, num_classes=16):
        super(TNet3, self).__init__()
        
        # Load the pre-trained VGG16 model
        self.vgg = models.vgg16(pretrained=True)
        
        # Modify the first convolutional layer to accept 1-channel input if necessary
        self.vgg.features[0] = nn.Conv2d(1, 64, kernel_size=3, stride=1, padding=1)
        
        # Freeze all layers initially
        for param in self.vgg.parameters():
            param.requires_grad = False
        
        # Unfreeze some layers
        for param in self.vgg.features[0:24].parameters():
            param.requires_grad = True
        
        # Replace the final fully connected layer to match the number of classes
        self.vgg.classifier[6] = nn.Sequential(
            nn.Dropout(0.5),
            nn.Linear(self.vgg.classifier[6].in_features, num_classes)
        )
        
        # Ensure the final layer's parameters are trainable
        for param in self.vgg.classifier[6].parameters():
            param.requires_grad = True

    def forward(self, x):
        x = self.vgg(x)
        return x

#--------------------------------------------------
#       Define Ensemble Model
#--------------------------------------------------

class EnsembleModel(nn.Module):
    def __init__(self, modelA, modelB, modelC):
        super(EnsembleModel, self).__init__()
        self.modelA = modelA
        self.modelB = modelB
        self.modelC = modelC

    def forward(self, x):
        outputA = self.modelA(x)
        outputB = self.modelB(x)
        outputC = self.modelC(x)
        return (outputA + outputB + outputC) / 3

# Initialize the individual models
modelA = TNet(num_classes=16)
modelB = TNet2(num_classes=16)
modelC = TNet3(num_classes=16)

# Create the ensemble model
ensemble_model = EnsembleModel(modelA, modelB, modelC)

#--------------------------------------------------
#       Model Training Function
#--------------------------------------------------

def trainModel(net, trainloader, train_option, testloader=None):
  loss_func = nn.CrossEntropyLoss()
  lr = train_option['lr']
  epoch = train_option['epoch']
  device = train_option['device'] if 'device' in train_option.keys() else 'cpu'
  log_iter = train_option['log_iter'] if 'log_iter' in train_option.keys() else 20
  eval_epoch = 1

  if 'optimizer' in train_option.keys():
    optimizer = train_option['optimizer']
  else:
    optimizer = optim.Adam(net.parameters(), lr=lr)
    
  scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=7, gamma=0.1)

  start_time = time.time()
  if device == 'gpu':
    net = net.cuda()

  iters = 0
  running_loss = 0.0
  for ep in range(epoch):
    net.train()
    for iter, (x, y) in enumerate(trainloader):
      iters += 1
      batch_x = Variable(x).float()
      batch_y = Variable(y).long()
      if device == 'gpu':
        batch_x = batch_x.cuda()
        batch_y = batch_y.cuda()

      outputs = net(batch_x)
      loss = loss_func(outputs, batch_y)
      optimizer.zero_grad()
      loss.backward()
      optimizer.step()
      running_loss += loss.item()

      time_lapse = time.strftime('%H:%M:%S', time.gmtime(time.time() - start_time))
      if iter % log_iter == 0:
        print('Epoch:{:2d} | Learning Rate: {} | Iter:{:5d} | Time: {} | Train Loss: {:.4f} | Average Loss: {:.4f} '.format(ep+1, scheduler.get_last_lr()[0],iter, time_lapse, loss.item(), running_loss/iters))

    scheduler.step()
    
    if testloader is not None and ep % eval_epoch == 0:
      evalModel(net, testloader)

#--------------------------------------------------
#       Model Evaluating Function
#--------------------------------------------------

def evalModel(net, testloader):
  acc = 0.0
  count = 0
  start_time = time.time()
  device = 'gpu' if next(net.parameters()).is_cuda else 'cpu'
  net.eval()

  for iter, (x, y) in enumerate(testloader):
        count += x.shape[0]
        batch_x = Variable(x).float()
        batch_y = Variable(y).long()
        if device == 'gpu':
          batch_x = batch_x.cuda()
          batch_y = batch_y.cuda()
        outputs = net(batch_x)
        acc += torch.sum(outputs.max(1)[1]==batch_y)

  time_lapse = time.strftime('%H:%M:%S', time.gmtime(time.time() - start_time))
  print('Accuracy: {:5f} | Time: {}'.format(acc/count,time_lapse))

#--------------------------------------------------
#       Start Training & Evaluation
#--------------------------------------------------
net = TNet()
net2 = TNet2()
train_option = {}
train_option['epoch'] = 15
train_option['device'] = 'gpu'

# Test with learning rate 0.001
train_option['lr'] = 0.001
print("Training with learning rate 0.001")
trainModel(modelA, trainloader_small, train_option, testloader_small)
