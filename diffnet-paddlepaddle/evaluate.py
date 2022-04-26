import math
import numpy as np

def getIdcg(length):
        idcg = 0.0
        for i in range(length):
            idcg = idcg + math.log(2) / math.log(i + 2)
        return idcg

def getDcg(value):
    dcg = math.log(2) / math.log(value + 2)
    return dcg

def getHr(value):
    hit = 1.0
    return hit

def evaluateRankingPerformance(evaluate_index_dict, evaluate_real_rating_matrix, \
    evaluate_predict_rating_matrix, topK, num_procs, exp_flag=0, sp_name=None, result_file=None):
    user_list = list(evaluate_index_dict.keys())
    batch_size = int(len(user_list) / num_procs)

    hr_list, ndcg_list = [], []
    index = 0
    for _ in range(num_procs):
        if index + batch_size < len(user_list):
            batch_user_list = user_list[index:index+batch_size]
            index = index + batch_size
        else:
            batch_user_list = user_list[index:len(user_list)]
        tmp_hr_list, tmp_ndcg_list = getHrNdcgProc(evaluate_index_dict, evaluate_real_rating_matrix, \
            evaluate_predict_rating_matrix, topK, batch_user_list)
        hr_list.extend(tmp_hr_list)
        ndcg_list.extend(tmp_ndcg_list)
    return np.mean(hr_list), np.mean(ndcg_list)

def getHrNdcgProc(
    evaluate_index_dict, 
    evaluate_real_rating_matrix,
    evaluate_predict_rating_matrix, 
    topK, 
    user_list):

    tmp_hr_list, tmp_ndcg_list = [], []

    for u in user_list:
        real_item_index_list = evaluate_index_dict[u]

        real_item_rating_list = np.concatenate(evaluate_real_rating_matrix[real_item_index_list]).tolist()
        positive_length = len(real_item_rating_list)
        target_length = min(positive_length, topK)
        
        predict_rating_list = evaluate_predict_rating_matrix[u]
        real_item_rating_list.extend(predict_rating_list)
        sort_index = np.argsort(real_item_rating_list)
        sort_index = sort_index[::-1]
        
        user_hr_list = []
        user_ndcg_list = []
        hits_num = 0
        for idx in range(topK):
            ranking = sort_index[idx]
            if ranking < positive_length:
                hits_num += 1
                user_hr_list.append(getHr(idx))
                user_ndcg_list.append(getDcg(idx))

        idcg = getIdcg(target_length)

        tmp_hr = np.sum(user_hr_list) / target_length
        tmp_ndcg = np.sum(user_ndcg_list) / idcg
        tmp_hr_list.append(tmp_hr)
        tmp_ndcg_list.append(tmp_ndcg)

    return tmp_hr_list, tmp_ndcg_list

# start evaluate model performance, hr and ndcg
def getPositivePredictions(conf, model, d_test_eva):
    d_test_eva.getEvaPositiveBatch()
    d_test_eva.linkedRankingEvaMap()
    
    user_list, item_list = d_test_eva.data_dict['EVA_USER_LIST'], d_test_eva.data_dict['EVA_ITEM_LIST']
    positive_predictions = model(user_list, item_list).numpy()
    return positive_predictions

def getNegativePredictions(conf, model, d_test_eva):
    negative_predictions = {}
    terminal_flag = 1
    while terminal_flag:
        batch_user_list, terminal_flag = d_test_eva.getEvaRankingBatch()
        d_test_eva.linkedRankingEvaMap()

        user_list, item_list = d_test_eva.data_dict['EVA_USER_LIST'], d_test_eva.data_dict['EVA_ITEM_LIST']
        tmp_negative_predictions = model(user_list, item_list).numpy().reshape(len(batch_user_list), -1)

        for index, u in enumerate(batch_user_list):
            negative_predictions[u] = tmp_negative_predictions[index]
    return negative_predictions


def evaluate(conf, model, d_test_eva):
    index_dict = d_test_eva.eva_index_dict
    positive_predictions = getPositivePredictions(conf, model, d_test_eva)
    negative_predictions = getNegativePredictions(conf, model, d_test_eva)

    d_test_eva.index = 0 # !!!important, prepare for new batch
    hr, ndcg = evaluateRankingPerformance(\
        index_dict, positive_predictions, negative_predictions, conf['topk'], conf['num_procs'])

    return hr, ndcg