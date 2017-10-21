# -*- coding: utf-8 -*-
"""
Created on Sat Aug 19 17:25:38 2017

@author: Kunal
"""
import numpy as np
from random import randint
import math
import random
population=[]
D=10;
def random_str():
    s=""
    for i in range(D*8):
        s=s+str(randint(0,1))
    return s   

def calc_value(stri):
    s=0
    sum=0
    for i in range(1,D+1):
        ans=int(stri[s+1:s+8],2)
        #print(ans)
        if stri[s]=='0':
            sum=sum+math.pow(10,6*((i-1)/(D-1)))*(math.pow(ans,2))
        else:
            sum=sum+math.pow(10,6*((i-1)/(D-1)))*(math.pow(-ans,2))
        s=s+8
    return sum
    


population_size=(D+1)*10
for i in range(population_size):
    population.append(random_str())



generation_size=D*100
min=math.pow(10,15)
for i in range(generation_size):
    #print(population)
    values=[]
    fitness=[]   
    # Calculate Value array
    for j in range(population_size):
        values.append(calc_value(population[j]))
    
    # Find Minimum value obtained so far
    for j in range(population_size):
        if values[j] < min:
            min=values[j]
    
    flag=1
    if min==0:
        flag=2
        print('result found in '+str(i+1)+' th generation')
        break;
    if flag==2:
        break;
    
    # Find Fitness Array
    if min==0:
        print(population)
        print(values)
        break
    for j in range(population_size):
        fitness.append(int(math.sqrt(100000000000/values[j])))
    mating_pool=[]
    
    
   
    print('avg in '+str(i+1)+' th generation: '+str(np.mean(fitness)))
    # Create Mating Pool
    for j in range(population_size):
        for k in range(fitness[j]):
            mating_pool.append(population[j])
        
    
    # Crossover
    new=0
    for j in range(int(population_size/2)):
        
        a=randint(0,len(mating_pool)-2)
        b=randint(0,len(mating_pool)-2)
        parentA=mating_pool[a]
        parentB=mating_pool[b]
        #print(parentA)
        #print(parentB)
        flag=1
        if min==0:
            flag=2
            print('result found in '+str(i+1)+' th generation')
            break;
        if flag==2:
            break;
        parentA=list(parentA)
        parentB=list(parentB)
        cross_site1=randint(0,int(((D*8)-1)/2) )
        cross_site2=randint(int(((D*8)-1)/2 + 1),D*8-1 )
        #print(cross_site1)
        #print(cross_site2)
        temp=parentA[cross_site1 : cross_site2+1]
        parentA[cross_site1 : cross_site2+1]=parentB[cross_site1 : cross_site2+1]
        parentB[cross_site1 : cross_site2+1]=temp
        #print(parentA)
        #print(parentB)
        
        #MUTATION
        mutation_rate=0.02
        for k in range(D*8):
            x=random.uniform(0,1)
            if(x < mutation_rate):
                if parentA[k]=='0':
                    parentA[k]='1'
                else:
                    parentA[k]='0'
        
        for k in range(D*8):
            x=random.uniform(0,1)
            if(x < mutation_rate):
                if parentB[k]=='0':
                    parentB[k]='1'
                else:
                    parentB[k]='0'
                    
        parentA=''.join(parentA)
        parentB=''.join(parentB)
        population[new]=parentA
        new=new+1
        population[new]=parentB
        new=new+1
    
    #print(values)
    #print(min)
    #print(fitness)
    #print(mating_pool)
    #print(len(mating_pool))
    #print('end')
    
print(min)