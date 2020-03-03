'''
Created on 19 Nov 2017

@author: Simon
'''

class DiagnosticModule(object):
    
    def __init__(self, funpointer, thaw_slump):
        self.Fun = funpointer
        self.thaw_slump = thaw_slump
        
    def evaluate(self):
        return self.Fun(self.thaw_slump)