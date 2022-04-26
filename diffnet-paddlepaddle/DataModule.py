'''
    author: Peijie Sun
    e-mail: sun.hfut@gmail.com 
    released date: 04/18/2019
'''
import pgl

from collections import defaultdict
import numpy as np
from time import time
import random

class DataModule():
    def __init__(self, conf, filename):
        self.conf = conf
        self.data_dict = {}
        self.terminal_flag = 1
        self.filename = filename
        self.index = 0

###########################################  Initalize Procedures ############################################
    def prepareGraphData(self):
        self.buildUserItemEdges()
        self.buildUserUserEdges()

        info_graph = pgl.Graph(num_nodes=self.conf['num_users']+self.conf['num_items'], edges=self.user_item_edges)
        soc_graph = pgl.Graph(num_nodes=self.conf['num_users'], edges=self.user_user_edges)

        return info_graph, soc_graph

    def initializeRankingTrain(self):
        self.readData()
        self.arrangePositiveData()
        self.generateTrainNegative()

    def initializeRankingVT(self):
        self.readData()
        self.arrangePositiveData()
        self.generateTrainNegative()

    def initalizeRankingEva(self):
        self.readData()
        self.getEvaPositiveBatch()
        self.generateEvaNegative()

    def linkedMap(self):
        self.data_dict['USER_LIST'] = self.user_list
        self.data_dict['ITEM_LIST'] = self.item_list
        self.data_dict['LABEL_LIST'] = self.labels_list
    
    def linkedRankingEvaMap(self):
        self.data_dict['EVA_USER_LIST'] = self.eva_user_list
        self.data_dict['EVA_ITEM_LIST'] = self.eva_item_list

###########################################  Ranking ############################################
    def readData(self):
        f = open(self.filename) ## May should be specific for different subtasks
        total_user_list = set()
        hash_data = defaultdict(int)
        for _, line in enumerate(f):
            arr = line.split("\t")
            hash_data[(int(arr[0]), int(arr[1]))] = 1
            total_user_list.add(int(arr[0]))
        self.total_user_list = list(total_user_list)
        self.hash_data = hash_data
    
    def arrangePositiveData(self):
        positive_data = defaultdict(set)
        total_data = set()
        hash_data = self.hash_data
        for (u, i) in hash_data:
            total_data.add((u, i))
            positive_data[u].add(i)
        self.positive_data = positive_data
        self.total_data = len(total_data)
    
    '''
        This function designes for the train/val/test negative generating section
    '''
    def generateTrainNegative(self):
        num_items = self.conf['num_items']
        num_negatives = self.conf['num_negatives']
        negative_data = defaultdict(set)
        total_data = set()
        hash_data = self.hash_data
        for (u, i) in hash_data:
            total_data.add((u, i))
            for _ in range(num_negatives):
                j = np.random.randint(num_items)
                while (u, j) in hash_data:
                    j = np.random.randint(num_items)
                negative_data[u].add(j)
                total_data.add((u, j))
        self.negative_data = negative_data
        self.terminal_flag = 1
        
    '''
        This function designes for the val/test section, compute loss
    '''
    def getVTRankingOneBatch(self):
        positive_data = self.positive_data
        negative_data = self.negative_data
        total_user_list = self.total_user_list
        user_list = []
        item_list = []
        labels_list = []
        for u in total_user_list:
            user_list.extend([u] * len(positive_data[u]))
            item_list.extend(positive_data[u])
            labels_list.extend([1] * len(positive_data[u]))
            user_list.extend([u] * len(negative_data[u]))
            item_list.extend(negative_data[u])
            labels_list.extend([0] * len(negative_data[u]))
        
        self.user_list = np.reshape(user_list, [-1, 1])
        self.item_list = np.reshape(item_list, [-1, 1])
        self.labels_list = np.reshape(labels_list, [-1, 1])
    
    '''
        This function designes for the training process
    '''
    def getTrainRankingBatch(self):
        positive_data = self.positive_data
        negative_data = self.negative_data
        total_user_list = self.total_user_list
        index = self.index
        batch_size = self.conf['training_batch_size']

        user_list, item_list, labels_list = [], [], []
        
        if index + batch_size < len(total_user_list):
            target_user_list = total_user_list[index:index+batch_size]
            self.index = index + batch_size
        else:
            target_user_list = total_user_list[index:len(total_user_list)]
            self.index = 0
            self.terminal_flag = 0

        for u in target_user_list:
            user_list.extend([u] * len(positive_data[u]))
            item_list.extend(list(positive_data[u]))
            labels_list.extend([1] * len(positive_data[u]))
            user_list.extend([u] * len(negative_data[u]))
            item_list.extend(list(negative_data[u]))
            labels_list.extend([0] * len(negative_data[u]))
        
        self.user_list = np.reshape(user_list, [-1, 1])
        self.item_list = np.reshape(item_list, [-1, 1])
        self.labels_list = np.reshape(labels_list, [-1, 1])
    
    '''
        This function designes for the positive data in rating evaluate section
    '''
    def getEvaPositiveBatch(self):
        hash_data = self.hash_data
        user_list = []
        item_list = []
        index_dict = defaultdict(list)
        index = 0
        for (u, i) in hash_data:
            user_list.append(u)
            item_list.append(i)
            index_dict[u].append(index)
            index = index + 1
        self.eva_user_list = np.reshape(user_list, [-1, 1])
        self.eva_item_list = np.reshape(item_list, [-1, 1])
        self.eva_index_dict = index_dict

    '''
        This function designes for the negative data generation process in rating evaluate section
    '''
    def generateEvaNegative(self):
        hash_data = self.hash_data
        total_user_list = self.total_user_list
        num_evaluate = self.conf['num_evaluate']
        num_items = self.conf['num_items']
        eva_negative_data = defaultdict(list)
        for u in total_user_list:
            for _ in range(num_evaluate):
                j = np.random.randint(num_items)
                while (u, j) in hash_data:
                    j = np.random.randint(num_items)
                eva_negative_data[u].append(j)
        self.eva_negative_data = eva_negative_data

    '''
        This function designs for the rating evaluate section, generate negative batch
    '''
    def getEvaRankingBatch(self):
        batch_size = self.conf['evaluate_batch_size']
        num_evaluate = self.conf['num_evaluate']
        eva_negative_data = self.eva_negative_data
        total_user_list = self.total_user_list
        index = self.index
        terminal_flag = 1
        total_users = len(total_user_list)
        user_list = []
        item_list = []
        if index + batch_size < total_users:
            batch_user_list = total_user_list[index:index+batch_size]
            self.index = index + batch_size
        else:
            terminal_flag = 0
            batch_user_list = total_user_list[index:total_users]
            self.index = 0
        for u in batch_user_list:
            user_list.extend([u]*num_evaluate)
            item_list.extend(eva_negative_data[u])
        self.eva_user_list = np.reshape(user_list, [-1, 1])
        self.eva_item_list = np.reshape(item_list, [-1, 1])
        return batch_user_list, terminal_flag

##################################################### Supplement for Sparse Computation ############################################
    def buildUserItemEdges(self):
        user_item_edges = []
        
        for (user, item) in self.hash_data.keys():
            user_item_edges.append((user, item+self.conf['num_users']))

        self.user_item_edges = user_item_edges

    def buildUserUserEdges(self, friends_flag=1):
        user_user_edges = []

        links_file = open(self.conf['links_filename'])
        for _, line in enumerate(links_file):
            tmp = line.split('\t')
            u1, u2 = int(tmp[0]), int(tmp[1])
            user_user_edges.append((u1, u2))
            if friends_flag == 1:
                user_user_edges.append((u2, u1))
        self.user_user_edges = user_user_edges

    
