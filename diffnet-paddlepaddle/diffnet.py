'''
    author: Peijie Sun
    e-mail: sun.hfut@gmail.com
    released date: 09/27/2021
'''

import paddle
import paddle.nn as nn
import paddle.nn.functional as F

import pgl
from pgl.nn import functional as GF

import numpy as np

class CustomGCNConv(nn.Layer):
    def __init__(self, input_size, output_size, graph):
        super(CustomGCNConv, self).__init__()
        self.input_size = input_size
        self.output_size = output_size
    
        self.graph = graph.tensor()

    def forward(self, feature):
        indegree = paddle.unsqueeze(self.graph.indegree(), axis=1) + 1e-12
        output = self.graph.send_recv(feature, 'sum')

        output = output * 1.0 / indegree

        output = output + feature 
        return output

class DiffNet(nn.Layer):
    def __init__(self, conf, info_graph, soc_graph, user_feature, item_feautre):
        super(DiffNet, self).__init__()

        self.conf = conf
        self.user_feature = paddle.to_tensor(user_feature)
        self.item_feature = paddle.to_tensor(item_feautre)

        # the user-item interactions form the infomation graph => info_graph
        self.infomation_gcn_layer = CustomGCNConv(self.conf['gnn_dim'], self.conf['gnn_dim'], info_graph)
        # the user-user relations form the social graph => soc_graph
        self.social_gcn_layer = CustomGCNConv(self.conf['gnn_dim'], self.conf['gnn_dim'], soc_graph)

        self.user_embedding = nn.Embedding(self.conf['num_users'], self.conf['gnn_dim'], sparse=True)
        self.item_embedding = nn.Embedding(self.conf['num_items'], self.conf['gnn_dim'], sparse=True)

        # initialize user_embedding and item_embedding from \mathcal{N}(\mu, \sigma^2)
        # self.user_embedding.weight.set_value(0.1 * np.random.randn(self.conf['num_users'], self.conf['gnn_dim']))
        # self.item_embedding.weight.set_value(0.1 * np.randn(self.conf['num_items'], self.conf['gnn_dim']))

        self.reduce_dim_layer = nn.Linear(self.conf['review_feature_dim'], self.conf['gnn_dim'])

        self.mse_loss = nn.MSELoss()

    def init_nodes_feature(self):
        first_user_feature = self.convertDistribution(self.user_feature)
        first_item_feature = self.convertDistribution(self.item_feature)

        second_user_feature = self.reduce_dim_layer(first_user_feature)
        second_item_feature = self.reduce_dim_layer(first_item_feature)

        third_user_feature = self.convertDistribution(second_user_feature)
        third_item_feature = self.convertDistribution(second_item_feature)

        self.init_user_feature = third_user_feature + self.user_embedding.weight
        self.init_item_feature = third_item_feature

    def convertDistribution(self, x):
        mean, std = paddle.mean(x), paddle.std(x)
        y = (x - mean) * 0.2 / std
        return y.astype('float32')

    def forward(self, user, item):
        user, item = paddle.to_tensor(user), paddle.to_tensor(item)

        self.init_nodes_feature()

        user_embedding_from_consumed_items = self.infomation_gcn_layer(paddle.concat([self.init_user_feature, self.init_item_feature], axis=0))[: self.conf['num_users']]

        first_gcn_user_embedding = self.social_gcn_layer(self.init_user_feature)
        second_gcn_user_embedding = self.social_gcn_layer(self.init_user_feature)

        # get the item embedding
        final_item_embed = paddle.index_select(self.item_embedding.weight + self.init_item_feature, item)

        final_user_embed = paddle.index_select(user_embedding_from_consumed_items + second_gcn_user_embedding, user)

        # predict ratings from user to item
        prediction = paddle.nn.functional.sigmoid(paddle.sum(final_user_embed * final_item_embed, axis=1, keepdim=True))

        return prediction
            

    def calculate_loss(self, user, item, label):
        prediction = self.forward(user, item)

        label = paddle.to_tensor(label, dtype='float32')
        loss = self.mse_loss(prediction, label)
        
        return loss