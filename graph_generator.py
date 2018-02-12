import numpy as np
import population
import matplotlib.pyplot as plt
import pylab as pl
import matplotlib.pyplot as plt
#plt.show()
#pl.ion()
def GenerateGraph():
    #xdata = []
    #ydata = []
    #plt.ion()
    #ax = plt.gca()
    #ax.set_autoscale_on(True)
    #ax.set_color_cycle(['red', 'black', 'yellow'])
    #line, = ax.plot(xdata, ydata,'o')
    #plt.axis([0, 1.5, 0, 1.5])
    file_ob=open("./log_folder/log_for_graph.txt", "r+")
    stlis = file_ob.readlines()
    stlislis = [ item.rstrip().split(' ') for item in stlis ]
    print(stlislis)
    numlislis = [[float(i) for i in item] for item in stlislis]
    newfront = np.array(numlislis)
    print(newfront)
    plt.plot(newfront[:,0], newfront[:,1], 'o')
    plt.axis([0, 1, 180, 420])
    plt.show()
    
    
    
GenerateGraph()