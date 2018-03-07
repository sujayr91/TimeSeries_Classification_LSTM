import h5py
import sys
import numpy as np
import torch
import torch.nn as nn
from torch.autograd import Variable
import argparse
import dill
from logutils import logger
from pdb import set_trace as st

class dataloader(object):
    '''
        Loads data from hdf5 file
    '''
    def __init__(self, datapath = None, dataset= 'train',batch_size = 64):
        self.datapath = datapath
        self.batch_size= batch_size
        self.data = h5py.File(datapath, 'r+')
        self.dataset = dataset + 'samples'
        self.datalabels = dataset + 'labels'
        self.totalsamples = self.data[self.dataset].shape[0]
        self.rowlength = self.data[self.dataset].shape[1]
        self.collength = self.data[self.dataset].shape[2]
        self.total_batches = int(self.totalsamples / self.batch_size)
        self.index = 0

    def __iter__(self):
        return(self)

    def __next__(self):
        if(self.index >=(self.totalsamples-self.batch_size + 1)):
            raise StopIteration
        batchdata = torch.FloatTensor(self.data[self.dataset][self.index:self.index+ self.batch_size, :])
        batchlabel = torch.LongTensor(self.data[self.datalabels][self.index:self.index+ self.batch_size])
        self.index = self.index+ self.batch_size
        return batchdata,batchlabel

    def __len__(self):
        return(self.total_batches)
   

class timeNet(nn.Module):
    '''
    model for timeseries classification
    '''
    def __init__(self, num_layers, input_size, hidden_size, num_classes):
        super(timeNet, self).__init__()
        self.lstm= nn.LSTM(input_size,hidden_size,num_layers, batch_first=True)
        self.linear = nn.Linear(hidden_size, num_classes)

    def init_weights(self):
        self.linear.weight.data.uniform_(-0.1,0.1)
        self.linear.bias.data.fill_(0)

    def forward(self,batch_input):
        out,_ = self.lstm(batch_input)
        out = self.linear(out[:,-1, :])  #Extract outout of lasttime step
        return out


parser = argparse.ArgumentParser()
parser.add_argument('--learning_rate', default = 0.01,type=float)
parser.add_argument('--max_epochs', default = 50,type=int)
parser.add_argument('--batch_size', default = 15,type=int)
parser.add_argument('--dataset_path', default ='./timeseriesdatabase.h5')
parser.add_argument('--checkpoint_path', default ='./checkpoint/')

logs = {}
logs['train_loss'], logs['test_loss'], logs['test_acc'] = logger(),logger(),logger() 
args = parser.parse_args()
logfile = './traininglogs_learningrate_{}.dill'.format(args.learning_rate)
epochs = args.max_epochs

def test(model, criterion):
    global logs
    testdata = dataloader(args.dataset_path, 'test', args.batch_size)
    correct = 0
    total = 0
    acc = 0
    testloss = 0
    num_batches = len(testdata)
    print(num_batches)
    for index, (inputs, targets) in enumerate(testdata):
       inputs= Variable(inputs)
       targets= Variable(targets)
       outputs = model(inputs)
       loss = criterion(outputs, targets)
       testloss+=loss.data[0]
       _,predicted=torch.max(outputs.data,1)
       correct += predicted.eq(targets.data).cpu().sum()
       total+=targets.size(0)
       print("In test loop")
    
    acc = 100.* correct/total
    avgtestloss = testloss/num_batches
    print("Test Accuracy: {} Test Loss: {}".format(acc, avgtestloss))
    return (acc, avgtestloss)


def train():
    global epochs
    global logs
    traindata = dataloader(args.dataset_path,'train', args.batch_size)
    model = timeNet(2, 18, 32, 3)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.SGD(model.parameters(), lr= args.learning_rate)
    num_batches =  len(traindata)
    print('total batches : {}'.format(num_batches))
    for epoch in range(0, epochs):
        for index, (inputs,targets) in enumerate(traindata):
            inputs = Variable(inputs)
            targets = Variable(targets)
            outputs= model(inputs)
            loss = criterion(outputs, targets)
            itrepoch = epoch + 1.* ((index + 1)/num_batches)
            logs['train_loss'].update(loss,itrepoch)
            if((index+1)% 5 == 0):
                print('Epoch = {}, Loss = {}'.format(epoch, loss.data[0]))
            if((index + 1)% 2 == 0):
                print('----------Starting Test ----------------')
                model.eval()
                testacc, testloss = test(model, criterion)
                model.train()
                logs['test_loss'].update(testloss, itrepoch)
                logs['test_acc'].update(testacc, itrepoch)

        if((epoch + 1)% 5 ==0):
            torch.save(model.state_dict(), args.checkpoint_path + 'checkpoint_epoch_{}.pth'.format(epoch))
        with open(logfile, 'wb') as mylogger:
            dill.dump(logs, mylogger)


if __name__=='__main__':
    train()
