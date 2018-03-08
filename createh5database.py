import numpy as np
import time
import timeit
from random import shuffle
import h5py
import os
import pickle

class H5Database():
    '''
        Class for creating HDF5 dataset
        Params:
        hdf5path: path to output hdf5 file
        datashape: training datashape, (n_rows, n_colms)
        buffer_size: hdf5 buffer length for write
        train_length: total training samples
        test_length : total test samples 
    '''
    def __init__(self, hdf5path,data_shape,buffer_size,train_length, test_length):
        self.db = h5py.File(hdf5path, mode = 'w')
        self.buffer_size = buffer_size
        self.train_length = train_length
        self.test_length = test_length
        self.sample_shape = data_shape
        self.train_db = None
        self.test_db = None
        self.train_label_db = None
        self.test_label_db = None
        self.idxs={"index": 0}
        self.sample_buffer = []
        self.label_buffer =[]
        self.create_hdf5_datasets()
        self.datasettype = "train"
    
    def initialize_dataset(self, datasettype):
        self.idxs['index'] = 0
        self.datasettype = datasettype

    def flushbuffers(self):
        if(self.datasettype == 'train'):
            self._write_buffer(self.train_db, self.sample_buffer)
            self._write_buffer(self.train_label_db, self.label_buffer)
        else:
            self._write_buffer(self.test_db, self.sample_buffer)
            self._write_buffer(self.test_label_db, self.label_buffer)
        self._clean_buffers()
    
    def close(self):
        print("Closing database")
        self.db.close()

    def create_hdf5_datasets(self):
        '''
            create train and test datasets
        '''
        ROWS, COLS = self.sample_shape
        self.train_label_db = self.db.create_dataset("trainlabels",(self.train_length,),maxshape = None, dtype ="int")
        self.test_label_db = self.db.create_dataset("testlabels",(self.test_length,),maxshape = None, dtype = "int")
        self.train_db = self.db.create_dataset("trainsamples", shape = (self.train_length, ROWS, COLS), dtype = "float")
        self.test_db = self.db.create_dataset("testsamples", shape = (self.test_length, ROWS, COLS), dtype = "float")

    def add(self, label, sample):
        '''
            Add samples to buffer. Write buffer to hdf5 when full
        '''
        self.sample_buffer.append(sample)
        self.label_buffer.append(label)
        if(len(self.sample_buffer)==self.buffer_size):
            if(self.datasettype == 'train'):
                self._write_buffer(self.train_db,self.sample_buffer)
                self._write_buffer(self.train_label_db, self.label_buffer)
            else:
                print('Write buffer............')
                print(self.label_buffer)
                self._write_buffer(self.test_db, self.sample_buffer)
                self._write_buffer(self.test_label_db, self.label_buffer)

            self.idxs['index']= self.idxs['index']+ len(self.label_buffer)
            self._clean_buffers()

    def _write_buffer(self, dataset, buf):
        '''
            write samples to buffer
        '''
        
        print("Writing Buffer {}".format(dataset))
        start = self.idxs['index']
        end = start+ len(buf)
        dataset[start:end] = buf


    def _clean_buffers(self):
        '''
            clear buffers
        '''
        self.sample_buffer =[]
        self.label_buffer =[]


def getlabel(label):
    '''
        Equivalent class labels based on datafolder where sample is pickeddd
    '''
    if(label=="Drowsy"):
        return 0
    elif(label == "Distracted"):
        return 1
    else:
        return 2

def builddatabase(datasamples, databasepath):
    '''
        create hdf5 database
        Input: 
        list of paths to datasamples in csv, database path

    '''
    features= [2,3,4,5,6,9,29,32,33,34,35,37,38,42,43,44,45,46]
    timelength = 256
    trainsplit = int(len(datasamples) * 0.75)
    traindatasamples = datasamples[:trainsplit]
    testdatasamples = datasamples[trainsplit:]
    dataset = H5Database(databasepath,(timelength,len(features)), 
            25, len(traindatasamples), 
            len(testdatasamples))
    train_mean = np.zeros((1,len(features)),dtype=float)
    sample_count =0
    features_sum = np.zeros((1,len(features)),dtype=float)
    print('######## Creating Train Dataset #########')
    dataset.initialize_dataset("train")
    for index,sample in enumerate(traindatasamples):
        print('Working on file:{} Complete : {:.2f}'.format(sample, 100.*index/len(traindatasamples)))
        try:
            data = np.genfromtxt(sample, delimiter=",")[-timelength:,features]
            features_sum+=data.sum(0)
            sample_count+=data.shape[0]
            train_mean = features_sum/(sample_count)
            labelname = sample.split("/")[2]
            label = getlabel(labelname)
            dataset.add(label,data)
        except:
            print('Exception in processing : {}'.format(sample))
            continue
       
    np.save('train_mean', train_mean)
    dataset.flushbuffers()
     
    print('######## Creating Test Dataset #########')
    dataset.initialize_dataset("test")
    for index,sample in enumerate(testdatasamples):
        print('Working on file:{} Complete : {:.2f}'.format(sample, 100.*index/len(testdatasamples)))
        try:
            data = np.genfromtxt(sample, delimiter=",")[-timelength:,features]
            labelname = sample.split("/")[2]
            label = getlabel(labelname)
            dataset.add(label,data)

        except:
            print('Exception in processing : {}'.format(sample))
            continue
        
    dataset.flushbuffers()
    dataset.close() 

if __name__=='__main__':
    datasamples =[]
    database = 'timeseriesdatabase.h5'
    rootpath = './dataset'
    datapaths = os.listdir('./dataset')
    for datapath in datapaths:
        currentpath = os.path.join(rootpath, datapath)
        for sample in os.listdir(currentpath):
            samplepath = os.path.join(currentpath,sample)
            datasamples.append(samplepath)
    shuffle(datasamples)    
    builddatabase(datasamples, database)
