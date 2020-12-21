import os, shutil
import ConfigParser as cp

class Logging():
    def __init__(self, filename):
        self.filename = filename
    
    def record(self, str_log):
        filename = self.filename
        print(str_log)
        with open(filename, 'a') as f:
            f.write("%s\r\n" % str_log)
            f.flush()
