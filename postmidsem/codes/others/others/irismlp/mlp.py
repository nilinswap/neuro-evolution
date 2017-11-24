
# Code from Chapter 4 of Machine Learning: An Algorithmic Perspective (2nd Edition)
# by Stephen Marsland (http://stephenmonika.net)

# You are free to use, change, or redistribute the code in any way you wish for
# non-commercial purposes, but please maintain the name of the original author.
# This code comes with no warranty of any kind.

# Stephen Marsland, 2008, 2014

import numpy as np
import pylab as pl
#import trainednetwork
class mlp:
    """ A Multi-Layer Perceptron"""
    
    def __init__(self,inputs,targets,nhidden,beta=1,momentum=0.9,outtype='logistic',weights1=None,weights2=None):
        
        """ Constructor """
        # Set up network size
        self.nin = np.shape(inputs)[1]
        self.nout = np.shape(targets)[1]
        self.ndata = np.shape(inputs)[0]
        self.nhidden = nhidden

        self.beta = beta
        self.momentum = momentum
        self.outtype = outtype
    
        # Initialise network
        #below is to intialize if weights are not given in parameter...see it is because I can't use 'self' inside above parenthesis
        if not weights1:
            weights1=(np.random.rand(self.nin+1,self.nhidden)-0.5)*2/np.sqrt(self.nin)
        if not weights2:
            weights2=(np.random.rand(self.nhidden+1,self.nout)-0.5)*2/np.sqrt(self.nhidden)
        self.weights1 = weights1
        self.weights2 = weights2

    def earlystopping(self,inputs,targets,valid,validtargets,eta,niterations=100,errcaltype='squaremean'):
        """here I thought it is not ready for more than one output nodes,
        because it expects validout and validtargets as one D array ( and not 2D array ). I was wrong.. sum takes care of my concern
        because otherwise I would find the mean of errors(for input set) seperately"""
        #valid = np.concatenate((valid,-np.ones((np.shape(valid)[0],1))),axis=1)
        #valid is changed before calling mlpfwd
        old_val_error1 = 100002
        old_val_error2 = 100001
        new_val_error = 100000
        
        count = 0
        lis=[]
        while (((old_val_error1 - new_val_error) > 0.001) or ((old_val_error2 - old_val_error1)>0.001)):
            count+=1
            print ("in earlystopping whileloop for "+str(count)+"th time")
            self.mlptrain(inputs,targets,eta,niterations)
            if errcaltype=='squaremean':
                meantrainerr=self.findmeantrainerr(inputs,targets)
            else:
                meantrainerr=self.confmat(inputs,targets)
            old_val_error2 = old_val_error1
            old_val_error1 = new_val_error
            if errcaltype=='squaremean':
                validout = self.mlpfwd(valid)
                new_val_error = 0.5*np.sum((validtargets-validout)**2)
            else:
                new_val_error=self.confmat(valid,validtargets)
            lis.append((meantrainerr,new_val_error))
        x=np.array([[i] for i in range(count)])
        t=np.array([[lis[i][0]] for i in range(count)])
        u=np.array([[lis[i][1]] for i in range(count)])
        """pl.plot(x,t,'o')
        pl.plot(x,u,'o')
        #pl.plot(x,net.mlpfwd(y),'o')
        pl.xlabel('no. of iteration\nblue is training error and yellow is validation error')
        pl.ylabel('error')
        pl.show() """  
        #above is used to plot a graph of error vs no. of iterations
        """a notable point is for some set of  'no. of nodes' , count was small so graph
        smaller;which means that for those no. of nodes, performance was just great
        See, there is one graph for one earlystopping i.e. for one value of no. of nodes in 
        network talking in reference to sinewave.py"""
        print ("Stopped")#, new_val_error,old_val_error1, old_val_error2)
        return new_val_error
    	
    def mlptrain(self,inputs,targets,eta,niterations):
        """ Train the thing """    
        # Add the inputs that match the bias node
        inputs = np.concatenate((inputs,-np.ones((self.ndata,1))),axis=1)
        change = range(self.ndata)
        updatew1 = np.zeros((np.shape(self.weights1)))
        updatew2 = np.zeros((np.shape(self.weights2)))
        for n in range(niterations):
            self.outputs = self.mlpfwd(inputs)
            error = 0.5*np.sum((self.outputs-targets)**2)
            if (np.mod(n,100)==0):
                print ("Iteration: ",n, " Error: ",error)    

            # Different types of output neurons
            if self.outtype == 'linear':
            	deltao = (self.outputs-targets)/self.ndata
            elif self.outtype == 'logistic':
            	deltao = self.beta*(self.outputs-targets)*self.outputs*(1.0-self.outputs)
            elif self.outtype == 'softmax':
                deltao = (self.outputs-targets)*(self.outputs*(-self.outputs)+self.outputs)/self.ndata 
            else:
            	print ("error")
            
            deltah = self.hidden*self.beta*(1.0-self.hidden)*(np.dot(deltao,np.transpose(self.weights2)))
                      
            updatew1 = eta*(np.dot(np.transpose(inputs),deltah[:,:-1])) + self.momentum*updatew1
            updatew2 = eta*(np.dot(np.transpose(self.hidden),deltao)) + self.momentum*updatew2
            self.weights1 -= updatew1
            self.weights2 -= updatew2
        
                
            # Randomise order of inputs (not necessary for matrix-based calculation)
            #np.random.shuffle(change)
            #inputs = inputs[change,:]
            #targets = targets[change,:]
            
    def mlpfwd(self,inputs):
        """ Run the network forward """
        #print("inputs",inputs)
        
        if np.shape(inputs)[1]==np.shape(self.weights1)[0]-1:
            inputs = np.concatenate((inputs,-np.ones((np.shape(inputs)[0],1))),axis=1)
        self.hidden = np.dot(inputs,self.weights1);
        self.hidden = 1.0/(1.0+np.exp(-self.beta*self.hidden))
        self.hidden = np.concatenate((self.hidden,-np.ones((np.shape(inputs)[0],1))),axis=1)

        outputs = np.dot(self.hidden,self.weights2);

        # Different types of output neurons
        if self.outtype == 'linear':
        	return outputs
        elif self.outtype == 'logistic':
            return 1.0/(1.0+np.exp(-self.beta*outputs))
        elif self.outtype == 'softmax':
            normalisers = np.sum(np.exp(outputs),axis=1)*np.ones((1,np.shape(outputs)[0]))
            return np.transpose(np.transpose(np.exp(outputs))/normalisers)
        else:
            print ("error")
    def findmeantrainerr(self,inputs,targets):
        newinputs = np.concatenate((inputs,-np.ones((self.ndata,1))),axis=1)
        newinputsoutput=self.mlpfwd(newinputs)
        meantrainerr=0.5*np.sum((targets-newinputsoutput)**2)
        return meantrainerr
    def confmat(self,inputs,targets):
        """Confusion matrix"""

        # Add the inputs that match the bias node
        inputs = np.concatenate((inputs,-np.ones((np.shape(inputs)[0],1))),axis=1)
        outputs = self.mlpfwd(inputs)
        
        nclasses = np.shape(targets)[1]

        if nclasses==1:
            nclasses = 2
            outputs = np.where(outputs>0.5,1,0)
        else:
            # 1-of-N encoding
            outputs = np.argmax(outputs,1)
            targets = np.argmax(targets,1)

        cm = np.zeros((nclasses,nclasses))
        for i in range(nclasses):
            for j in range(nclasses):
                cm[i,j] = np.sum(np.where(outputs==i,1,0)*np.where(targets==j,1,0))

        print (cm)
        err= 1-(np.trace(cm)/np.sum(cm)) 
        print(err)
        return err
