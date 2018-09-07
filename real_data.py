import numpy as np
import csv

class RealIds():
    
    def __init__(self):
        self.ids = []
        self.cursor = 0
        self.wordLenLimit = 40
        self.load()
        self.vectorSize1D = len(self.all_chars) * self.wordLenLimit
        
    def load(self):
        chars ={}
        reader = csv.reader(open('real_ids.csv', newline=''), delimiter=' ', quotechar='|')
        for row in reader:
            self.ids.append(row[0])
            id = row[0]
            for char in list(id):
                chars[char] = True

        # print(ids)
        self.all_chars = list(chars.keys())
        self.all_chars.sort()        
        self.all_chars = [''] + self.all_chars
        self.iMat = np.identity(len(self.all_chars))
        

    def wordToMatrix(self, word):
        # print("len(self.all_chars)", len(self.all_chars))
        mat = np.zeros([self.wordLenLimit, len(self.all_chars)])
        i = 0
        offset = np.random.randint(0, self.wordLenLimit-len(word)) 
        for char in list(word):
            i = i+1
            idx = self.all_chars.index(char)
            mat[offset+i, : ] = self.iMat[idx]
        return mat
    
    def matrixToWord(self, mat):
        idxses = np.argmax(mat, axis=1)        
        chars = []
        for idx in idxses:            
            chars.append(self.all_chars[idx])
        return ''.join(chars)

    def get_batch(self, num):    
        items = []
        if self.cursor + num > len(self.ids):
            items = self.ids[self.cursor:]
            items = items + self.ids[0 : (self.cursor+num) % len(self.ids)]
            self.cursor = (self.cursor+num) % len(self.ids)
        else:
            items = self.ids[self.cursor:self.cursor+num]
            self.cursor = self.cursor+num
        return np.array([self.wordToMatrix(x) for x in items])
        


