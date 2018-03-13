import gc
import h5py
import sys
import numpy as np
import torch
import torch.nn as nn
from torch.autograd import Variable
import argparse
import dill
from logutils import logger
import numpy.matlib as repmat
from pdb import set_trace as st
class dataloader(object):
    '''
        Loads data from hdf5 file
    '''
    def __init__(self, datapath = None, dataset= 'train',mean_path = './train_mean.npy', batch_size = 64):
        self.datapath = datapath
        self.batch_size= batch_size
        self.data = h5py.File(datapath, 'r+')
        self.dataset = dataset + 'samples'
        self.datalabels = dataset + 'labels'
        self.totalsamples = self.data[self.dataset].shape[0]
        self.rowlength = self.data[self.dataset].shape[1]
        self.collength = self.data[self.dataset].shape[2]
        self.total_batches = int(self.totalsamples / self.batch_size)
        self.train_mean =  np.load(mean_path)
        assert self.train_mean.shape[1] == self.collength, "Mean feature length, datasamples feature length doesn't match"
        self.train_mean = repmat.repmat(self.train_mean, self.batch_size,1)
        self.train_mean = repmat.repmat(self.train_mean,1,self.rowlength).reshape(self.batch_size,self.rowlength, self.collength)
        self.index = 0

    def __iter__(self):
        return(self)

    def __next__(self):
        if(self.index >=(self.totalsamples-self.batch_size + 1)):
            self.index = 0
            raise StopIteration
        batchdata = self.data[self.dataset][self.index:self.index+ self.batch_size, :]
        batchdata-=self.train_mean
        batchdata = torch.FloatTensor(batchdata)
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
parser.add_argument('--learning_rate', default = 0.0001,type=float)
parser.add_argument('--max_epochs', default = 50,type=int)
parser.add_argument('--batch_size', default = 15,type=int)
parser.add_argument('--dataset_path', default ='./timeseriesdatabase.h5')
parser.add_argument('--mean_path', default ='./train_mean.npy')
parser.add_argument('--checkpoint_path', default ='./checkpoint/')

logs = {}
logs['train_loss'], logs['test_loss'], logs['test_acc'] = logger(),logger(),logger() 
args = parser.parse_args()
logfile = './traininglogs_learningrate_{}.dill'.format(args.learning_rate)
epochs = args.max_epochs

def test(model, criterion):
    global logs
    testdata = dataloader(args.dataset_path, 'test', args.mean_path, args.batch_size)
    correct = 0
    total = 0
    acc = 0
    testloss = 0
    num_batches = len(testdata)
    print(num_batches)
    for index, (features, targets) in enumerate(testdata):
       inputs= Variable(features)
       targets= Variable(targets)
       outputs = model(inputs)
       loss = criterion(outputs, targets)
       testloss+=loss.data[0]
       _,predicted=torch.max(outputs.data,1)
       correct += predicted.eq(targets.data).cpu().sum()
       total+=targets.size(0)
       del features, inputs, outputs, targets
       gc.collect()
    
    acc = 100.* correct/total
    avgtestloss = testloss/num_batches
    print("Test Accuracy: {} Test Loss: {}".format(acc, avgtestloss))
    return (acc, avgtestloss)
@profile
def train():
    global epochs
    global logs
    traindata = dataloader(args.dataset_path,'train',args.mean_path, args.batch_size)
    model = timeNet(2, 18, 32, 3)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.SGD(model.parameters(), lr= args.learning_rate)
    num_batches =  len(traindata)
    print('total batches : {}'.format(num_batches))
    for epoch in range(0, epochs):
        for index, (features,targets) in enumerate(traindata):
            inputs = Variable(features)
            targets = Variable(targets)
            outputs= model(inputs)
            loss = criterion(outputs, targets)
            loss.backward()
            optimizer.step()
            itrepoch = epoch + 1.* ((index + 1)/num_batches)
            logs['train_loss'].update(loss.data[0],itrepoch)
            if((index+1)% 5 == 0):
                print('Epoch = {}, Loss = {}'.format(epoch, loss.data[0]))
            del inputs,outputs, targets,features,loss
            gc.collect()

        
        print('----------Starting Test ----------------')
        model.eval()
        testacc, testloss = test(model, criterion)
        model.train()
        logs['test_loss'].update(testloss, epoch)
        logs['test_acc'].update(testacc, epoch)

        if((epoch + 1)% 5 ==0):
            torch.save(model.state_dict(), args.checkpoint_path + 'checkpoint_epoch_{}.pth'.format(epoch))
        with open(logfile, 'wb') as mylogger:
            dill.dump(logs, mylogger)


if __name__=='__main__':
    train()
