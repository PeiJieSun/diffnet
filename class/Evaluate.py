import math
import numpy as np


class Evaluate():
    def __init__(self, conf):
        self.conf = conf

    def getIdcg(self, length):
        idcg = 0.0
        for i in range(length):
            idcg = idcg + math.log(2) / math.log(i + 2)
        return idcg

    def getDcg(self, value):
        dcg = math.log(2) / math.log(value + 2)
        return dcg

    def getHr(self, value):
        hit = 1.0
        return hit

    def evaluateRankingPerformance(self, evaluate_index_dict, evaluate_real_rating_matrix, \
        evaluate_predict_rating_matrix, topK, num_procs, exp_flag=0, sp_name=None, result_file=None):
        user_list = list(evaluate_index_dict.keys())
        batch_size = len(user_list) / num_procs

        hr_list, ndcg_list = [], []
        index = 0
        for _ in range(num_procs):
            if index + batch_size < len(user_list):
                batch_user_list = user_list[index:index+batch_size]
                index = index + batch_size
            else:
                batch_user_list = user_list[index:len(user_list)]
            tmp_hr_list, tmp_ndcg_list = self.getHrNdcgProc(evaluate_index_dict, evaluate_real_rating_matrix, \
                evaluate_predict_rating_matrix, topK, batch_user_list)
            hr_list.extend(tmp_hr_list)
            ndcg_list.extend(tmp_ndcg_list)
        return np.mean(hr_list), np.mean(ndcg_list)

    def getHrNdcgProc(self, 
        evaluate_index_dict, 
        evaluate_real_rating_matrix,
        evaluate_predict_rating_matrix, 
        topK, 
        user_list):

        tmp_hr_list, tmp_ndcg_list = [], []

        for u in user_list:
            real_item_index_list = evaluate_index_dict[u]
            real_item_rating_list = list(np.concatenate(evaluate_real_rating_matrix[real_item_index_list]))
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
                    user_hr_list.append(self.getHr(idx))
                    user_ndcg_list.append(self.getDcg(idx))

            idcg = self.getIdcg(target_length)

            tmp_hr = np.sum(user_hr_list) / target_length
            tmp_ndcg = np.sum(user_ndcg_list) / idcg
            tmp_hr_list.append(tmp_hr)
            tmp_ndcg_list.append(tmp_ndcg)

        return tmp_hr_list, tmp_ndcg_list

    def evaluateRankingPerformance_sparsity(self, evaluate_index_dict, social_sparsity_dict, interest_sparsity_dict, evaluate_real_rating_matrix, \
        evaluate_predict_rating_matrix, topK, num_procs, exp_flag=0, sp_name=None, result_file=None):
        user_list = list(evaluate_index_dict.keys())
        batch_size = len(user_list) / num_procs

        social_hr_list_0_4, social_ndcg_list_0_4 = [], []
        social_hr_list_4_8, social_ndcg_list_4_8 = [], []
        social_hr_list_8_16, social_ndcg_list_8_16 = [], []
        social_hr_list_16_32, social_ndcg_list_16_32 = [], []
        social_hr_list_32_64, social_ndcg_list_32_64 = [], []
        social_hr_list_64, social_ndcg_list_64 = [], []
        


        interest_hr_list_0_4, interest_ndcg_list_0_4 = [], []
        interest_hr_list_4_8, interest_ndcg_list_4_8 = [], []
        interest_hr_list_8_16, interest_ndcg_list_8_16 = [], []
        interest_hr_list_16_32, interest_ndcg_list_16_32 = [], []
        interest_hr_list_32_64, interest_ndcg_list_32_64 = [], []
        interest_hr_list_64, interest_ndcg_list_64 = [], []



        index = 0
        for _ in range(num_procs):
            if index + batch_size < len(user_list):
                batch_user_list = user_list[index:index+batch_size]
                index = index + batch_size
            else:
                batch_user_list = user_list[index:len(user_list)]
            #set_trace()
            #social
            social_tmp_hr_list_0_4, social_tmp_ndcg_list_0_4,\
            social_tmp_hr_list_4_8, social_tmp_ndcg_list_4_8,\
            social_tmp_hr_list_8_16, social_tmp_ndcg_list_8_16,\
            social_tmp_hr_list_16_32, social_tmp_ndcg_list_16_32,\
            social_tmp_hr_list_32_64, social_tmp_ndcg_list_32_64, \
            social_tmp_hr_list_64, social_tmp_ndcg_list_64  = self.getHrNdcgProc_social_sparsity(evaluate_index_dict, social_sparsity_dict, interest_sparsity_dict, evaluate_real_rating_matrix, \
                evaluate_predict_rating_matrix, topK, batch_user_list)

            social_hr_list_0_4.extend(social_tmp_hr_list_0_4)
            social_ndcg_list_0_4.extend(social_tmp_ndcg_list_0_4)
            social_hr_list_4_8.extend(social_tmp_hr_list_4_8)
            social_ndcg_list_4_8.extend(social_tmp_ndcg_list_4_8)
            social_hr_list_8_16.extend(social_tmp_hr_list_8_16)
            social_ndcg_list_8_16.extend(social_tmp_ndcg_list_8_16)
            social_hr_list_16_32.extend(social_tmp_hr_list_16_32)
            social_ndcg_list_16_32.extend(social_tmp_ndcg_list_16_32)
            social_hr_list_32_64.extend(social_tmp_hr_list_32_64)
            social_ndcg_list_32_64.extend(social_tmp_ndcg_list_32_64)
            social_hr_list_64.extend(social_tmp_hr_list_64)
            social_ndcg_list_64.extend(social_tmp_ndcg_list_64)


            #interest
            interest_tmp_hr_list_0_4, interest_tmp_ndcg_list_0_4,\
            interest_tmp_hr_list_4_8, interest_tmp_ndcg_list_4_8,\
            interest_tmp_hr_list_8_16, interest_tmp_ndcg_list_8_16,\
            interest_tmp_hr_list_16_32, interest_tmp_ndcg_list_16_32,\
            interest_tmp_hr_list_32_64, interest_tmp_ndcg_list_32_64, \
            interest_tmp_hr_list_64, interest_tmp_ndcg_list_64  = self.getHrNdcgProc_interest_sparsity(evaluate_index_dict, social_sparsity_dict, interest_sparsity_dict, evaluate_real_rating_matrix, \
                evaluate_predict_rating_matrix, topK, batch_user_list)

            interest_hr_list_0_4.extend(interest_tmp_hr_list_0_4)
            interest_ndcg_list_0_4.extend(interest_tmp_ndcg_list_0_4)
            interest_hr_list_4_8.extend(interest_tmp_hr_list_4_8)
            interest_ndcg_list_4_8.extend(interest_tmp_ndcg_list_4_8)
            interest_hr_list_8_16.extend(interest_tmp_hr_list_8_16)
            interest_ndcg_list_8_16.extend(interest_tmp_ndcg_list_8_16)
            interest_hr_list_16_32.extend(interest_tmp_hr_list_16_32)
            interest_ndcg_list_16_32.extend(interest_tmp_ndcg_list_16_32)
            interest_hr_list_32_64.extend(interest_tmp_hr_list_32_64)
            interest_ndcg_list_32_64.extend(interest_tmp_ndcg_list_32_64)
            interest_hr_list_64.extend(interest_tmp_hr_list_64)
            interest_ndcg_list_64.extend(interest_tmp_ndcg_list_64)


        #set_trace()
        return np.sum(social_hr_list_0_4)/len(social_sparsity_dict['0-4']), np.sum(social_ndcg_list_0_4)/len(social_sparsity_dict['0-4']),\
               np.sum(social_hr_list_4_8)/len(social_sparsity_dict['4-8']), np.sum(social_ndcg_list_4_8)/len(social_sparsity_dict['4-8']),\
               np.sum(social_hr_list_8_16)/len(social_sparsity_dict['8-16']), np.sum(social_ndcg_list_8_16)/len(social_sparsity_dict['8-16']),\
               np.sum(social_hr_list_16_32)/len(social_sparsity_dict['16-32']), np.sum(social_ndcg_list_16_32)/len(social_sparsity_dict['16-32']),\
               np.sum(social_hr_list_32_64)/len(social_sparsity_dict['32-64']), np.sum(social_ndcg_list_32_64)/len(social_sparsity_dict['32-64']), \
               np.sum(social_hr_list_64)/len(social_sparsity_dict['64-']), np.sum(social_ndcg_list_64)/len(social_sparsity_dict['64-']), \
               np.sum(interest_hr_list_0_4)/len(interest_sparsity_dict['0-4']), np.sum(interest_ndcg_list_0_4)/len(interest_sparsity_dict['0-4']),\
               np.sum(interest_hr_list_4_8)/len(interest_sparsity_dict['4-8']), np.sum(interest_ndcg_list_4_8)/len(interest_sparsity_dict['4-8']),\
               np.sum(interest_hr_list_8_16)/len(interest_sparsity_dict['8-16']), np.sum(interest_ndcg_list_8_16)/len(interest_sparsity_dict['8-16']),\
               np.sum(interest_hr_list_16_32)/len(interest_sparsity_dict['16-32']), np.sum(interest_ndcg_list_16_32)/len(interest_sparsity_dict['16-32']),\
               np.sum(interest_hr_list_32_64)/len(interest_sparsity_dict['32-64']), np.sum(interest_ndcg_list_32_64)/len(interest_sparsity_dict['32-64']), \
               np.sum(interest_hr_list_64)/len(interest_sparsity_dict['64-']), np.sum(interest_ndcg_list_64)/len(interest_sparsity_dict['64-'])


    def getHrNdcgProc_social_sparsity(self, 
        evaluate_index_dict, 
        social_sparsity_dict,
        interest_sparsity_dict,
        evaluate_real_rating_matrix,
        evaluate_predict_rating_matrix, 
        topK, 
        user_list):

        social_tmp_hr_list_0_4, social_tmp_ndcg_list_0_4= [], []
        social_tmp_hr_list_4_8, social_tmp_ndcg_list_4_8= [], []
        social_tmp_hr_list_8_16, social_tmp_ndcg_list_8_16= [], []
        social_tmp_hr_list_16_32, social_tmp_ndcg_list_16_32= [], []
        social_tmp_hr_list_32_64, social_tmp_ndcg_list_32_64= [], []
        social_tmp_hr_list_64, social_tmp_ndcg_list_64= [], []
     

        for u in user_list:
            real_item_index_list = evaluate_index_dict[u]
            real_item_rating_list = list(np.concatenate(evaluate_real_rating_matrix[real_item_index_list]))
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
                    user_hr_list.append(self.getHr(idx))
                    user_ndcg_list.append(self.getDcg(idx))

            idcg = self.getIdcg(target_length)

            tmp_hr = np.sum(user_hr_list) / target_length
            tmp_ndcg = np.sum(user_ndcg_list) / idcg
            #set_trace()
            if( u in social_sparsity_dict['64-'] ):
                social_tmp_hr_list_64.append(tmp_hr)
                social_tmp_ndcg_list_64.append(tmp_ndcg)              
            elif( u in social_sparsity_dict['32-64'] ):
                social_tmp_hr_list_32_64.append(tmp_hr)
                social_tmp_ndcg_list_32_64.append(tmp_ndcg)
            elif( u in social_sparsity_dict['16-32'] ):
                social_tmp_hr_list_16_32.append(tmp_hr)
                social_tmp_ndcg_list_16_32.append(tmp_ndcg)  
            elif( u in social_sparsity_dict['8-16'] ):
                social_tmp_hr_list_8_16.append(tmp_hr)
                social_tmp_ndcg_list_8_16.append(tmp_ndcg)  
            elif( u in social_sparsity_dict['4-8'] ):
                social_tmp_hr_list_4_8.append(tmp_hr)
                social_tmp_ndcg_list_4_8.append(tmp_ndcg)  
            elif( u in social_sparsity_dict['0-4'] ):
                social_tmp_hr_list_0_4.append(tmp_hr)
                social_tmp_ndcg_list_0_4.append(tmp_ndcg)     


        return social_tmp_hr_list_0_4, social_tmp_ndcg_list_0_4, \
               social_tmp_hr_list_4_8, social_tmp_ndcg_list_4_8, \
               social_tmp_hr_list_8_16, social_tmp_ndcg_list_8_16, \
               social_tmp_hr_list_16_32, social_tmp_ndcg_list_16_32, \
               social_tmp_hr_list_32_64, social_tmp_ndcg_list_32_64, \
               social_tmp_hr_list_64, social_tmp_ndcg_list_64
              



    def getHrNdcgProc_interest_sparsity(self, 
        evaluate_index_dict, 
        social_sparsity_dict,
        interest_sparsity_dict,
        evaluate_real_rating_matrix,
        evaluate_predict_rating_matrix, 
        topK, 
        user_list):

        interest_tmp_hr_list_0_4, interest_tmp_ndcg_list_0_4= [], []
        interest_tmp_hr_list_4_8, interest_tmp_ndcg_list_4_8= [], []
        interest_tmp_hr_list_8_16, interest_tmp_ndcg_list_8_16= [], []
        interest_tmp_hr_list_16_32, interest_tmp_ndcg_list_16_32= [], []
        interest_tmp_hr_list_32_64, interest_tmp_ndcg_list_32_64= [], []
        interest_tmp_hr_list_64, interest_tmp_ndcg_list_64= [], []


        for u in user_list:
            real_item_index_list = evaluate_index_dict[u]
            real_item_rating_list = list(np.concatenate(evaluate_real_rating_matrix[real_item_index_list]))
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
                    user_hr_list.append(self.getHr(idx))
                    user_ndcg_list.append(self.getDcg(idx))

            idcg = self.getIdcg(target_length)

            tmp_hr = np.sum(user_hr_list) / target_length
            tmp_ndcg = np.sum(user_ndcg_list) / idcg
            #set_trace()
            if( u in interest_sparsity_dict['64-'] ):
                interest_tmp_hr_list_64.append(tmp_hr)
                interest_tmp_ndcg_list_64.append(tmp_ndcg)   
            elif( u in interest_sparsity_dict['32-64']):
                interest_tmp_hr_list_32_64.append(tmp_hr)
                interest_tmp_ndcg_list_32_64.append(tmp_ndcg) 
            elif( u in interest_sparsity_dict['16-32']):
                interest_tmp_hr_list_16_32.append(tmp_hr)
                interest_tmp_ndcg_list_16_32.append(tmp_ndcg) 
            elif( u in interest_sparsity_dict['8-16']):
                interest_tmp_hr_list_8_16.append(tmp_hr)
                interest_tmp_ndcg_list_8_16.append(tmp_ndcg)             
            elif( u in interest_sparsity_dict['4-8'] ):
                interest_tmp_hr_list_4_8.append(tmp_hr)
                interest_tmp_ndcg_list_4_8.append(tmp_ndcg)   
            elif( u in interest_sparsity_dict['0-4']):
                interest_tmp_hr_list_0_4.append(tmp_hr)
                interest_tmp_ndcg_list_0_4.append(tmp_ndcg)

        return interest_tmp_hr_list_0_4, interest_tmp_ndcg_list_0_4, \
               interest_tmp_hr_list_4_8, interest_tmp_ndcg_list_4_8, \
               interest_tmp_hr_list_8_16, interest_tmp_ndcg_list_8_16, \
               interest_tmp_hr_list_16_32, interest_tmp_ndcg_list_16_32, \
               interest_tmp_hr_list_32_64, interest_tmp_ndcg_list_32_64, \
               interest_tmp_hr_list_64, interest_tmp_ndcg_list_64
              









