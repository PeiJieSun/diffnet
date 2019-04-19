import ConfigParser as cp
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

    def parserConf(self, platform_config='', data_name='', model_name=''):
        conf = cp.ConfigParser()
        conf.read(self.config_path)
        self.conf = conf

        self.conf_dict = {}

        for section in conf.sections():
            for (key, value) in conf.items(section):
                self.processValue(key, value)

        # Following fours parameters are the common ones in all experiments
        if platform_config != '':
            self.model_name = model_name
            self.data_name = data_name
            self.root_dir = os.path.join(platform_config.app_log_dir, data_name)
            # Ex: /home/sunpeijie/files/task/pyrec/data/dual_amazon_books/dual_amazon_books
            self.data_dir = os.path.join(platform_config.app_data_dir, data_name, data_name)