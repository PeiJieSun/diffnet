'''
    author: Peijie Sun
    e-mail: sun.hfut@gmail.com 
    released date: 04/18/2019
'''

import os
from time import time
from DataModule import DataModule

class DataUtil():
    def __init__(self, conf):
        self.conf = conf
        #print('DataUtil, Line12, test- conf data_dir:%s' % self.conf.data_dir)

    def initializeRankingHandle(self):
        #t0 = time()
        self.createTrainHandle()
        self.createEvaluateHandle()
        #t1 = time()
        #print('Prepare data cost:%.4fs' % (t1 - t0))
    
    def createTrainHandle(self):
        data_dir = self.conf.data_dir
        train_filename = "%s/%s.train.rating" % (data_dir, self.conf.data_name)
        val_filename = "%s/%s.val.rating" % (data_dir, self.conf.data_name)
        test_filename = "%s/%s.test.rating" % (data_dir, self.conf.data_name)

        self.train = DataModule(self.conf, train_filename)
        self.val = DataModule(self.conf, val_filename)
        self.test = DataModule(self.conf, test_filename)

    def createEvaluateHandle(self):
        data_dir = self.conf.data_dir
        val_filename = "%s/%s.val.rating" % (data_dir, self.conf.data_name)
        test_filename = "%s/%s.test.rating" % (data_dir, self.conf.data_name)

        self.val_eva = DataModule(self.conf, val_filename)
        self.test_eva = DataModule(self.conf, test_filename)
