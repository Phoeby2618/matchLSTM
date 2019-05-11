import numpy as np

from newdataProcess import Preprocess
import tensorflow.contrib.keras as kr


class BatchGenerator(object):
    '''
    construct raw_data generator.The input X,y should be narray or list like type
    '''
    def __init__(self,X1,X2,y,x1_len,x2_len,word_dict,shuffle):
        if type(X1)!=np.ndarray:
            X=np.array(X1)
        if type(y)!=np.ndarray:
            y=np.array(y)
        self.X1=X1
        self.X2=X2
        self.y=y
        self.x1_len=x1_len
        self.x2_len=x2_len
        self.word_dicts=word_dict
        self._epochs_completed=0
        self._index_in_epoch=0
        self._number_example=X1.shape[0]
        self.shuffle=shuffle
        if shuffle:
            index=np.random.permutation(self._number_example)
            self.X1=self.X1[index]
            self.X2=self.X2[index]
            self.x1_len=self.x1_len[index]
            self.x2_len = self.x2_len[index]
            self.y=self.y[index]

    def X(self):
            return self.X

    def y(self):
            return self.y

    def _epochs_completed(self):
            return self._epochs_completed

    def _number_example(self):
            return self._number_example

    def padding(self, sentences,max_len):

        sens=[]
        for s in sentences:
                num = max_len - len(s)

                s0 = s[:]
                for i in range(num):
                    s0.append(self.word_dicts['PADDING'])
                sens.append(s0)
        return sens

    def next_batch(self,batch_size):
            '''return raw_data in batch_size
                consider epoche
            '''
            start=self._index_in_epoch
            self._index_in_epoch+=batch_size
            if self._index_in_epoch>self._number_example:
                self._epochs_completed+=1
                if self.shuffle:
                    index = np.random.permutation(self._number_example)
                    self.X1 = self.X1[index]
                    self.X2 = self.X2[index]
                    self.x1_len = self.x1_len[index]
                    self.x2_len = self.x2_len[index]
                    self.y = self.y[index]

                start=0                                                      #这里这么写是因为又开始了新的batch，start=0开始
                self._index_in_epoch=batch_size
                assert batch_size<self._number_example
            end=self._index_in_epoch

            s1_len = self.x1_len[start:end]
            s2_len=self.x2_len[start:end]
            label=self.y[start:end]
            max_len = max(max(s1_len), max(s2_len))
            s1_random=self.X1[start:end]
            s2_random=self.X2[start:end]

            s1 = self.padding(list(s1_random), max_len)
            s2 = self.padding(list(s2_random), max_len)

            return s1,s2,label,s1_len,s2_len,max_len

if __name__=='__main__':
    file_dir2 = 'data/prosciTail/'
    file_name2 = 'scitail_dev.txt'
    word_dict_file = 'data/word_dict/sciTail_word_dict.pkl'

    # train=DataHelper(file_dir2,file_name2,word_dict_file,20,True)
    data = Preprocess(file_dir2, file_name2, word_dict_file)
    sentence1 = data.s1
    sentence2 = data.s2
    label = data.label
    sen1_length = data.s1_length
    sen2_length = data.s2_length
    word_dict = data.word_dict

    data_train = BatchGenerator(sentence1, sentence2, label, sen1_length, sen2_length, word_dict, True)

    for i in range(5):
        s1, s2, label, s1_len, s2_len, max_len = data_train.next_batch(20)
        print(s1)
        print(s1_len)
        print(s2)
        print(s2_len)
        print(label)
        print(max_len)
        sen1_len=[]
        for s in s1:
            sen1_len.append(len(s))
        print(sen1_len)