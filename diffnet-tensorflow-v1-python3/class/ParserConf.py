'''
    author: Peijie Sun
    e-mail: sun.hfut@gmail.com 
    released date: 04/18/2019
'''

import configparser as cp
import re, os

class ParserConf():

    def __init__(self, config_path):
        self.config_path = config_path

    def processValue(self, key, value):
        #print(key, value)
        tmp = value.split(' ')
        dtype = tmp[0]
        value = tmp[1:]
        #print(dtype, value)

        if value != None:
            if dtype == 'string':
                self.conf_dict[key] = vars(self)[key] = value[0]
            elif dtype == 'int':
                self.conf_dict[key] = vars(self)[key] = int(value[0])
            elif dtype == 'float':
                self.conf_dict[key] = vars(self)[key] = float(value[0])
            elif dtype == 'list':
                self.conf_dict[key] = vars(self)[key] = [i for i in value]
            elif dtype == 'int_list':
                self.conf_dict[key] = vars(self)[key] = [int(i) for i in value]
            elif dtype == 'float_list':
                self.conf_dict[key] = vars(self)[key] = [float(i) for i in value]
        else:
            print('%s value is None' % key)

    def parserConf(self):
        conf = cp.ConfigParser()
        conf.read(self.config_path)
        self.conf = conf

        self.conf_dict = {}
        for section in conf.sections():
            for (key, value) in conf.items(section):
                print(key, value)
                self.processValue(key, value)
       
        self.data_dir = os.path.join(os.getcwd(), 'data/%s' % self.data_name)
        self.links_filename = os.path.join(os.getcwd(), 'data/%s/%s.links' % (self.data_name, self.data_name))
        self.user_review_vector_matrix = os.path.join(os.getcwd(), 'data/%s/user_vector.npy' % self.data_name)
        self.item_review_vector_matrix = os.path.join(os.getcwd(), 'data/%s/item_vector.npy' % self.data_name)
        self.pre_model = os.path.join(os.getcwd(), 'pretrain/%s/%s' % (self.data_name, self.pre_model))
