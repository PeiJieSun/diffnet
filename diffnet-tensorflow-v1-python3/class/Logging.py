'''
    author: Peijie Sun
    e-mail: sun.hfut@gmail.com 
    released date: 04/18/2019
'''

import os, shutil
import configparser as cp

class Logging():
    def __init__(self, filename):
        self.filename = filename
    
    def record(self, str_log):
        filename = self.filename
        print(str_log)
        with open(filename, 'a') as f:
            f.write("%s\r\n" % str_log)
            f.flush()
