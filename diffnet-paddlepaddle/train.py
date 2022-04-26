import os

from time import time
from tqdm import tqdm

import paddle
import numpy as np

from DataUtil import DataUtil
from diffnet import DiffNet
from evaluate import evaluate
from config_diffnet import get_params

os.environ["CUDA_VISIBLE_DEVICES"]="0"


def main(conf):
    ############################## BASIC CONFIGURES ##############################
    data = DataUtil(conf)

    ############################## PREPARE DATASET ##############################
    print('System start to load data...')
    t0 = time()

    data.initializeRankingHandle()
    d_train, d_val, d_test, d_test_eva = data.train, data.val, data.test, data.test_eva
    
    d_train.initializeRankingTrain()
    d_val.initializeRankingVT()
    d_test.initializeRankingVT()
    d_test_eva.initalizeRankingEva()
    t1 = time()
    print('Data has been loaded successfully, cost:%.4fs' % (t1 - t0))

    ############################## CREATE MODEL ##############################
    info_graph, soc_graph = d_train.prepareGraphData()
    user_feature = np.load(conf['user_review_vector_matrix'], allow_pickle=True)
    item_feature = np.load(conf['item_review_vector_matrix'], allow_pickle=True)
    model = DiffNet(conf, info_graph, soc_graph, user_feature, item_feature)

    # define optimizer
    optimizer = paddle.optimizer.Adam(\
        learning_rate=conf['learning_rate'], parameters=model.parameters())
    
    ########################### START TRAINING & EVALUATION #####################################

    for epoch in range(1, conf['epochs']+1):
        t0 = time()

        model.train()
        train_loss_list = []

        while d_train.terminal_flag:
            d_train.getTrainRankingBatch()
            d_train.linkedMap()

            user_list, item_list, label_list = \
                d_train.data_dict['USER_LIST'], d_train.data_dict['ITEM_LIST'], d_train.data_dict['LABEL_LIST']
            
            loss = model.calculate_loss(user_list, item_list, label_list)
           
            #print('loss:%.4f' % loss.item())
            train_loss_list.append(loss.item())

            loss.backward()
            optimizer.step()
            optimizer.clear_grad()

        t1 = time()
        print('Epoch:{}, Training Stage: compute loss cost:{:.4f}s'.\
            format(epoch, (t1-t0)))
        print('Train loss:{:.6f}'.format(np.mean(train_loss_list)))

        # compute val loss
        d_val.getVTRankingOneBatch()
        d_val.linkedMap()

        user_list, item_list, label_list = \
                d_val.data_dict['USER_LIST'], d_val.data_dict['ITEM_LIST'], d_val.data_dict['LABEL_LIST']

        val_loss = model.calculate_loss(user_list, item_list, label_list)

        # compute test loss
        d_test.getVTRankingOneBatch()
        d_test.linkedMap()
        
        user_list, item_list, label_list = \
                d_test.data_dict['USER_LIST'], d_test.data_dict['ITEM_LIST'], d_test.data_dict['LABEL_LIST']

        test_loss = model.calculate_loss(user_list, item_list, label_list)

        t2 = time()
        print('Epoch:{}, Testing Stage: compute loss cost:{:.4f}s'.\
            format(epoch, (t2-t1)))
        print('Eva loss:{:.6f}, Test loss:{:.6f}'.format(val_loss.item(), test_loss.item()))
        
        # start to evaluate model performance via computing the hr and ndcg values of the model
        model.eval()
       
        t1 = time()
        hr, ndcg = evaluate(conf, model, d_test_eva)
        t2 = time()

        print('Evaluate cost:%.4fs, hr:%.4f, ndcg:%.4f' % ((t2-t1), hr, ndcg))

        ## reset train data pointer, and generate new negative data
        d_train.generateTrainNegative()

if __name__ == '__main__':
    conf = vars(get_params())
    main(conf)