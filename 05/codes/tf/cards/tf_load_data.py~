from __future__ import print_function

__docformat__ = 'restructedtext en'

import six.moves.cPickle as pickle
import gzip
import os
import sys
import timeit

import numpy as np


def load_data(dataset):

    def preprocessdata(dataset):
        path="/home/robita/forgit/neuro-evolution/05/dataset/"
        import os.path
        filename=path+"cards/"+dataset
        if  os.path.isfile(filename):
            fileo=open(filename,"r+")
            stlis=fileo.readlines()
            stlis=[i.rstrip().split(',') for i in stlis]

            mislis=[]
            indlis=[i for i in range(len(stlis)) if '?' in stlis[i]]
            tempstlis=stlis[:]
            for i in indlis:
                mislis.append(tempstlis[i])
                stlis.remove(tempstlis[i])
            del(tempstlis)


            
            featarr=np.array(stlis)

            #first feat starts
            featarr[np.where(featarr[:,0]=='a'),0]=0
            featarr[np.where(featarr[:,0]=='b'),0]=1
            # ends

            #second starts
            #ends

            #third starts
            #ends

            #fourth starts
            featarr[np.where(featarr[:,3]=='u'),3]=0
            featarr[np.where(featarr[:,3]=='y'),3]=1
            featarr[np.where(featarr[:,3]=='l'),3]=2
            featarr[np.where(featarr[:,3]=='t'),3]=3
            #fourth ends

            #fifth starts
            featarr[np.where(featarr[:,4]=='g'),4]=0
            featarr[np.where(featarr[:,4]=='p'),4]=1
            featarr[np.where(featarr[:,4]=='gg'),4]=2
            #fifth ends

            #sixth starts
            featvarlis=['c', 'd', 'cc', 'i', 'j', 'k', 'm', 'r', 'q', 'w', 'x', 'e', 'aa', 'ff']
            ite=range(len(featvarlis))
            for i in ite:
                featarr[np.where(featarr[:,5]==featvarlis[i]),5]=i
            #sixth ends

            #seventh starts
            featvarlis=['v', 'h', 'bb','j', 'n', 'z', 'dd', 'ff', 'o']
            ite=range(len(featvarlis))
            for i in ite:
                featarr[np.where(featarr[:,6]==featvarlis[i]),6]=i
            #that too ends

            #eighth starts
            #eighth ends

            #nineth starts
            featarr[np.where(featarr[:,8]=='f'),8]=0
            featarr[np.where(featarr[:,8]=='t'),8]=1
            #well this ends too

            #tenth starts
            featarr[np.where(featarr[:,9]=='f'),9]=0
            featarr[np.where(featarr[:,9]=='t'),9]=1
            #why not end this as well?

            #eleventh starts
            #and ends at the same place

            #twelweth starts
            featarr[np.where(featarr[:,11]=='f'),11]=0
            featarr[np.where(featarr[:,11]=='t'),11]=1
            #everything ends...except true love

            #thirteenth starts
            featarr[np.where(featarr[:,12]=='g'),12]=0
            featarr[np.where(featarr[:,12]=='p'),12]=1
            featarr[np.where(featarr[:,12]=='s'),12]=2
            #what's the matter with you?

            #You know what? I won't even mention forteenth and fifteenth

            featarr[np.where(featarr[:,15]=='-'),15]=0       
            featarr[np.where(featarr[:,15]=='+'),15]=1

            #normalizing
            #featarr[:,:15]=(featarr[:,:15]-np.mean(featarr[:,:15],axis=0))/(np.max(featarr[:,:15],axis=0)-np.min(featarr[:,:15],axis=0))

            featarr=featarr.astype(float)
            traindata=featarr
            means= traindata.mean(axis=0)

            stdevs=np.std(traindata,axis=0)

             
            # standardize dataset
            def standardize_dataset(traindata, means, stdevs):
                for row in traindata:
                    for i in range(len(row)):

                        row[i] = (row[i] - means[i])
                        if stdevs[i]:
                            row[i]/=stdevs[i]
            standardize_dataset(traindata[:,:15],means,stdevs)
            
            return featarr,mislis
        else:
            print("file could not be loaded")
   
    featarr,mislis=preprocessdata(dataset)#here mislis is list type and featar is nd.array type
    
    np.random.shuffle(featarr)

    test_set=featarr[:133,:15],featarr[:133,15]#keeping test set aside
    rest_set=featarr[133:,:15],featarr[133:,15]# have to apply cross-validation on the rest

    print("test",test_set[0].shape,"rest",rest_set[0].shape)

    

    rval = [test_set,rest_set,mislis]
    return rval

def main():
    load_data('cards.data')
if __name__=='__main__':
    main()