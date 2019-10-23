'''
    author: Peijie Sun
    e-mail: sun.hfut@gmail.com 
    released date: 04/18/2019
'''
import tensorflow as tf
import numpy as np

class diffnet():
    def __init__(self, conf):
        self.conf = conf
        self.supply_set = (
            'SOCIAL_NEIGHBORS_SPARSE_MATRIX',
            'CONSUMED_ITEMS_SPARSE_MATRIX'
        )

    def startConstructGraph(self):
        self.initializeNodes()
        self.constructTrainGraph()
        self.saveVariables()
        self.defineMap()

    def inputSupply(self, data_dict):
        self.social_neighbors_indices_input = data_dict['SOCIAL_NEIGHBORS_INDICES_INPUT']
        self.social_neighbors_values_input = data_dict['SOCIAL_NEIGHBORS_VALUES_INPUT']

        self.consumed_items_indices_input = data_dict['CONSUMED_ITEMS_INDICES_INPUT']
        self.consumed_items_values_input = data_dict['CONSUMED_ITEMS_VALUES_INPUT']

        # prepare sparse matrix, in order to compute user's embedding from social neighbors and consumed items
        self.social_neighbors_dense_shape = np.array([self.conf.num_users, self.conf.num_users]).astype(np.int64)
        self.consumed_items_dense_shape = np.array([self.conf.num_users, self.conf.num_items]).astype(np.int64)

        self.social_neighbors_sparse_matrix = tf.SparseTensor(
            indices = self.social_neighbors_indices_input, 
            values = self.social_neighbors_values_input,
            dense_shape=self.social_neighbors_dense_shape
        )
        self.consumed_items_sparse_matrix = tf.SparseTensor(
            indices = self.consumed_items_indices_input, 
            values = self.consumed_items_values_input,
            dense_shape=self.consumed_items_dense_shape
        )

    def convertDistribution(self, x):
        mean, var = tf.nn.moments(x, axes=[0, 1])
        y = (x - mean) * 0.2 / tf.sqrt(var)
        return y

    def generateUserEmbeddingFromSocialNeighbors(self, current_user_embedding):
        user_embedding_from_social_neighbors = tf.sparse_tensor_dense_matmul(
            self.social_neighbors_sparse_matrix, current_user_embedding
        )
        return user_embedding_from_social_neighbors
    
    def generateUserEmebddingFromConsumedItems(self, current_item_embedding):
        user_embedding_from_consumed_items = tf.sparse_tensor_dense_matmul(
            self.consumed_items_sparse_matrix, current_item_embedding
        )
        return user_embedding_from_consumed_items

    def initializeNodes(self):
        self.item_input = tf.placeholder("int32", [None, 1]) # Get item embedding from the core_item_input
        self.user_input = tf.placeholder("int32", [None, 1]) # Get user embedding from the core_user_input
        self.labels_input = tf.placeholder("float32", [None, 1])

        self.user_embedding = tf.Variable(
            tf.random_normal([self.conf.num_users, self.conf.dimension], stddev=0.01), name='user_embedding')
        self.item_embedding = tf.Variable(
            tf.random_normal([self.conf.num_items, self.conf.dimension], stddev=0.01), name='item_embedding')

        self.user_review_vector_matrix = tf.constant(\
            np.load(self.conf.user_review_vector_matrix), dtype=tf.float32)
        self.item_review_vector_matrix = tf.constant(\
            np.load(self.conf.item_review_vector_matrix), dtype=tf.float32)
        self.reduce_dimension_layer = tf.layers.Dense(\
            self.conf.dimension, activation=tf.nn.sigmoid, name='reduce_dimension_layer')

        self.item_fusion_layer = tf.layers.Dense(\
            self.conf.dimension, activation=tf.nn.sigmoid, name='item_fusion_layer')
        self.user_fusion_layer = tf.layers.Dense(\
            self.conf.dimension, activation=tf.nn.sigmoid, name='user_fusion_layer')

    def constructTrainGraph(self):
        # handle review information, map the origin review into the new space and 
        first_user_review_vector_matrix = self.convertDistribution(self.user_review_vector_matrix)
        first_item_review_vector_matrix = self.convertDistribution(self.item_review_vector_matrix)
        
        self.user_reduce_dim_vector_matrix = self.reduce_dimension_layer(first_user_review_vector_matrix)
        self.item_reduce_dim_vector_matrix = self.reduce_dimension_layer(first_item_review_vector_matrix)

        second_user_review_vector_matrix = self.convertDistribution(self.user_reduce_dim_vector_matrix)
        second_item_review_vector_matrix = self.convertDistribution(self.item_reduce_dim_vector_matrix)

        # compute item embedding
        #self.fusion_item_embedding = self.item_fusion_layer(\
        #   tf.concat([self.item_embedding, second_item_review_vector_matrix], 1))
        self.final_item_embedding = self.fusion_item_embedding \
                             = self.item_embedding + second_item_review_vector_matrix
        #self.final_item_embedding = self.fusion_item_embedding = second_item_review_vector_matrix

        # compute user embedding
        user_embedding_from_consumed_items = self.generateUserEmebddingFromConsumedItems(self.final_item_embedding)

        #self.fusion_user_embedding = self.user_fusion_layer(\
        #    tf.concat([self.user_embedding, second_user_review_vector_matrix], 1))
        self.fusion_user_embedding = self.user_embedding + second_user_review_vector_matrix
        first_gcn_user_embedding = self.generateUserEmbeddingFromSocialNeighbors(self.fusion_user_embedding)
        second_gcn_user_embedding = self.generateUserEmbeddingFromSocialNeighbors(first_gcn_user_embedding)

        self.final_user_embedding = second_gcn_user_embedding + user_embedding_from_consumed_items

        latest_user_latent = tf.gather_nd(self.final_user_embedding, self.user_input)
        latest_item_latent = tf.gather_nd(self.final_item_embedding, self.item_input)

        predict_vector = tf.multiply(latest_user_latent, latest_item_latent)
        self.prediction = tf.sigmoid(tf.reduce_sum(predict_vector, 1, keepdims=True))
        #self.prediction = self.predict_rating_layer(tf.concat([latest_user_latent, latest_item_latent], 1))

        self.loss = tf.nn.l2_loss(self.labels_input - self.prediction)

        self.opt_loss = tf.nn.l2_loss(self.labels_input - self.prediction)
        self.opt = tf.train.AdamOptimizer(self.conf.learning_rate).minimize(self.opt_loss)
        self.init = tf.global_variables_initializer()

    def saveVariables(self):
        ############################# Save Variables #################################
        variables_dict = {}
        variables_dict[self.user_embedding.op.name] = self.user_embedding
        variables_dict[self.item_embedding.op.name] = self.item_embedding

        for v in self.reduce_dimension_layer.variables:
            variables_dict[v.op.name] = v
                
        self.saver = tf.train.Saver(variables_dict)
        ############################# Save Variables #################################
    
    def defineMap(self):
        map_dict = {}
        map_dict['train'] = {
            self.user_input: 'USER_LIST', 
            self.item_input: 'ITEM_LIST', 
            self.labels_input: 'LABEL_LIST'
        }
        
        map_dict['val'] = {
            self.user_input: 'USER_LIST', 
            self.item_input: 'ITEM_LIST', 
            self.labels_input: 'LABEL_LIST'
        }

        map_dict['test'] = {
            self.user_input: 'USER_LIST', 
            self.item_input: 'ITEM_LIST', 
            self.labels_input: 'LABEL_LIST'
        }

        map_dict['eva'] = {
            self.user_input: 'EVA_USER_LIST', 
            self.item_input: 'EVA_ITEM_LIST'
        }

        map_dict['out'] = {
            'train': self.loss,
            'val': self.loss,
            'test': self.loss,
            'eva': self.prediction
        }

        self.map_dict = map_dict
