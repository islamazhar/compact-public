
import sys 
sys.path.append("..")


import torch
import torchvision
import torch.nn as nn
import torch.nn.functional as F

import os
import pickle


import matplotlib.pyplot as plt


def get_default_device():
    """Pick GPU if available, else CPU"""
    # return 'cpu'

    if torch.cuda.is_available():
        return torch.device('cuda')
    else:
        return torch.device('cpu')
    
def to_device(data, device):
    """Move tensor(s) to chosen device"""
    if isinstance(data, (list,tuple)):
        return [to_device(x, device) for x in data]
    return data.to(device, non_blocking=True)
tol = 1e-3
k_max = 10
class DeviceDataLoader():
    """Wrap a dataloader to move data to a device"""
    def __init__(self, dl, device):
        self.dl = dl
        self.device = device
        
    def __iter__(self):
        """Yield a batch of data after moving it to device"""
        for b in self.dl: 
            yield to_device(b, self.device)

    def __len__(self):
        """Number of batches"""
        return len(self.dl)
    
    
    
class ImageClassificationBase(nn.Module):
    def training_step(self, batch):
        images, labels = batch 
        out = self(images)                  # Generate predictions
        loss = F.cross_entropy(out, labels) # Calculate loss
        return loss
    

    def validation_step(self, batch):
        images, labels = batch 
        out = self(images)                    # Generate predictions
        loss = F.cross_entropy(out, labels)   # Calculate loss
        acc = accuracy(out, labels)           # Calculate accuracy
            
        return {'val_loss': loss.detach(), 'val_acc': acc}
        
    def validation_epoch_end(self, outputs):
        batch_losses = [x['val_loss'] for x in outputs]
        epoch_loss = torch.stack(batch_losses).mean()   # Combine losses
        batch_accs = [x['val_acc'] for x in outputs]
        epoch_acc = torch.stack(batch_accs).mean()      # Combine accuracies
        
        return {'val_loss': epoch_loss.item(), 'val_acc': epoch_acc.item()}
    
    def secure_validation_step(self, batch, mpcApproximation):
        images, labels = batch 
        out = self.secure_forward(images, mpcApproximation)   # Generate secure predictions
        loss = F.cross_entropy(out, labels)   # Calculate loss
        acc = accuracy(out, labels)           # Calculate accuracy
        return {'val_loss': loss, 'val_acc': acc}
    
    def epoch_end(self, epoch, result):
        print("Epoch [{}], train_loss: {:.4f}, val_loss: {:.4f}, val_acc: {:.4f}".format(
            epoch, result['train_loss'], result['val_loss'], result['val_acc']))
        
def accuracy(outputs, labels):
    _, preds = torch.max(outputs, dim=1)
    return torch.tensor(torch.sum(preds == labels).item() / len(preds))


normalize_vars = {}

class CnnModel(ImageClassificationBase):
    def __init__(self,in_channels=3, num_classes=10, act_func=nn.ReLU, mu=0, sigma=0, act_approx=None):
        self.activationFunc = act_func
        super().__init__()
        self.network = nn.Sequential(
            nn.Conv2d(in_channels, 32, kernel_size=3, padding=1), # in_channel = 3 out channel = 32
            nn.BatchNorm2d(32), # batch_size, num_channels, [2d input]
            self.activationFunc(),
            nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(64),
            self.activationFunc(),
            nn.MaxPool2d(2, 2), # output: 64 x 16 x 16

            nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(128),
            self.activationFunc(),
            nn.Conv2d(128, 128, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(128),
            self.activationFunc(),
            nn.MaxPool2d(2, 2), # output: 128 x 8 x 8

            nn.Conv2d(128, 256, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(256),
            self.activationFunc(),
            nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(256),
            self.activationFunc(),
            nn.MaxPool2d(2, 2), # output: 256 x 4 x 4

            nn.Flatten(), 
            nn.Linear(256*4*4, 1024),
            nn.BatchNorm1d(1024), # [investigate why 1D here?]
            self.activationFunc(),
            nn.Linear(1024, 512),
            nn.BatchNorm1d(512),
            self.activationFunc(), # can also add dropout here...
            nn.Linear(512, num_classes),
            # nn.LogSoftmax(dim=1) ?
        )
        self.mu, self.sigma = mu, sigma
        self.act_approx = act_approx
        
    def forward(self, xb):
        # print(xb.shape)
        for i in range(len(self.network)):
            # if "activation" in str(type(self.network[i])):
            #     xb = nn.ReLU()(xb)
            # else:            
                xb = self.network[i](xb) 
        return xb
    
    def secure_forward(self, xb, mpcApproximation):
        
        # xb = xb.view(xb.size(0), -1)
        for i in range(len(self.network)):
            if "activation" in str(type(self.network[i])):
                xb = mpcApproximation.predict(xb)
            else:
                xb = self.network[i](xb)
                 
        return xb
    
    
    
    
@torch.no_grad()
def evaluate(model, val_loader):
    model.eval() 
    outputs = [model.validation_step(batch) for batch in val_loader]
    return model.validation_epoch_end(outputs)

@torch.no_grad()
def secure_evaluate(model, val_loader, mpcApproximation):
    """ Evaluate the model's performance on the validation set securely using 
        MPC approximation.
    """
    model.eval()
    outputs = []    
    outputs = [model.secure_validation_step(batch, mpcApproximation) for batch in val_loader]
    return model.validation_epoch_end(outputs) # May need to change

def fit(epochs, lr, model, train_loader, val_loader, opt_func=torch.optim.SGD):
    history = []
    optimizer = opt_func(model.parameters(), lr)
    for epoch in range(epochs):
        # Training Phase 
        model.train()
        train_losses = []
        for batch in train_loader:
            loss = model.training_step(batch)
            train_losses.append(loss)
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()
        # Validation phase
        result = evaluate(model, val_loader)
        result['train_loss'] = torch.stack(train_losses).mean().item()
        model.epoch_end(epoch, result)
        history.append(result)
        print(f"Done with epoch = {epoch} and result = {result}")
    return history
        


def get_name(obj):
    return str(obj)[36:-2]



def plot_accuracies(histories):
    for key, history in histories.items():
        accuracies = [x['val_acc'] for x in history]
        plt.plot(accuracies, '-x', label=str(key)[36:-2])
    
    plt.xlabel('epoch')
    plt.ylabel('accuracy')
    plt.legend()
    plt.title('Accuracy vs. No. of epochs');
    
    
def plot_losses(histories):
    for key, history in histories.items():
        train_losses = [x.get('train_loss') for x in history]
        val_losses = [x['val_loss'] for x in history]
        plt.plot(train_losses, '-x', label=get_name(key))
        plt.plot(val_losses, '-x', label=get_name(key))
        
    plt.xlabel('epoch')
    plt.ylabel('loss')
    plt.legend()
    plt.title('Loss vs. No. of epochs')
    plt.show()
    
# plot_losses(histories)



