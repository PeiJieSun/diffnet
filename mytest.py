'''
    This script is used to evaluate the performance of the ranking models
'''
import os
import numpy as np
import tensorflow as tf
from Analysis import Analysis

print('Test flag!!')

def start(conf, data, model, evaluate, pre_model):
    #conf = ParserConf(config_file)
    #conf.parserConf()
    
     # second define constant and prepare evaluate data
    data.createEvaluateHandle()
    d_train = data.train
    test_eva = data.test_eva
    test_eva.initalizeRankingEva()

    # prepare model necessary data.
    if model.supply == True:
        d_train.initializeRankingTrain()
        data_dict = d_train.prepareModelSupplement(model)
        model.inputSupply(data_dict)
        model.startConstructGraph()

    # standard tensorflow running environment initialize
    tf_conf = tf.ConfigProto()
    os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3' 
    tf_conf.gpu_options.allow_growth = True
    sess = tf.Session(config=tf_conf)
    sess.run(model.init)

    model.saver.restore(sess, pre_model)

    # Start Evaluate !!!

    # start evaluate model performance, hr and ndcg
    def getPositivePredictions():
        test_eva.linkedRankingEvaMap()
        eva_feed_dict = {}
        for (key, value) in model.map_dict['eva'].items():
            eva_feed_dict[key] = test_eva.data_dict[value]
        positive_predictions = sess.run(
            model.map_dict['out']['eva'],
            feed_dict=eva_feed_dict)
        return positive_predictions

    def getNegativePredictions(eva):
        negative_predictions = {}
        terminal_flag = 1
        while terminal_flag:
            batch_user_list, terminal_flag = test_eva.getEvaRankingBatch()
            test_eva.linkedRankingEvaMap()
            eva_feed_dict = {}
            for (key, value) in model.map_dict['eva'].items():
                eva_feed_dict[key] = eva.data_dict[value]
            index = 0
            tmp_negative_predictions = np.reshape(
                sess.run(
                    model.map_dict['out']['eva'],
                    feed_dict=eva_feed_dict
                ),
                [-1, conf.num_evaluate])
            for u in batch_user_list:
                negative_predictions[u] = tmp_negative_predictions[index]
                index = index + 1
        return negative_predictions

    index_dict = test_eva.eva_index_dict
    positive_predictions = getPositivePredictions()
    negative_predictions = getNegativePredictions(test_eva)

    topK_list = [5, 10, 15]
    for topK in topK_list:
        hr, ndcg = evaluate.evaluateRankingPerformance(\
            index_dict, positive_predictions, negative_predictions, topK, conf.num_procs)
        print('topK:%d ----- hr:%.4f ----- ndcg:%.4f' % (topK, hr, ndcg))

    #### sparsity analysis ####
    neigh_dict = {}
    for line in open(conf.user_neigh):
        tmp = line.strip('\n').split('\t')
        u, num_neigh = int(tmp[0]), int(tmp[1])
        neigh_dict[u] = num_neigh
    topK = 5
    hr, ndcg, hr_dict, ndcg_dict = evaluate.sparseAnalysisHR_NDCG(\
            index_dict, positive_predictions, negative_predictions, topK)
    analysis = Analysis()
    user_list = list(index_dict.keys())
    print('user_list length:%d, max:%d, min:%d' % (len(user_list), max(user_list), min(user_list)))
    analysis.sparseAnalysis(conf.dataset, conf.model_name, user_list, neigh_dict, hr_dict, ndcg_dict)