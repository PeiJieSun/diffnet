'''
    author: Peijie Sun
    e-mail: sun.hfut@gmail.com 
    released date: 04/18/2019
'''

import sys, os, argparse

sys.path.append(os.path.join(os.getcwd(), 'class'))

from ParserConf import ParserConf
from DataUtil import DataUtil
from Evaluate import Evaluate

from diffnet import diffnet

def executeTrainModel(config_path, model_name):
    print(config_path)
    #print('System start to prepare parser config file...')
    conf = ParserConf(config_path)
    conf.parserConf()
    print conf.topk
    
    #print('System start to load TensorFlow graph...')
    model = eval(model_name)
    model = model(conf)

    #print('System start to load data...')
    data = DataUtil(conf)
    evaluate = Evaluate(conf)

    import train as starter
    starter.start(conf, data, model, evaluate)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Welcome to the Experiment Platform Entry')
    parser.add_argument('--data_name', nargs='?', help='data name')
    parser.add_argument('--model_name', nargs='?', help='model name')
    parser.add_argument('--gpu', nargs='?', help='available gpu id')

    args = parser.parse_args()

    data_name = args.data_name
    model_name = args.model_name
    device_id = args.gpu
    
    if args.gpu:
        os.environ['CUDA_VISIBLE_DEVICES'] = device_id
    config_path = os.path.join(os.getcwd(), 'conf/%s_%s.ini' % (data_name, model_name))

    executeTrainModel(config_path, model_name)
