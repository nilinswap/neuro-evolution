# -*- coding: utf-8 -*-
"""
Created on Fri Aug 25 00:06:39 2017

@author: Kunal
"""

import numpy as np
import math
import random
from random import randint
population=[]
D=10

def random_pop():
    x=[]
    for i in range(D):
       x.append(random.uniform(-100,100))
    return x
 
def calc_value(x):
    ans=0
    for i in range(D):
        ans=ans+x[i]*x[i]-10*math.cos(2*math.pi*x[i])+10
    return ans     

population_size=150
for i in range(population_size):
    population.append(random_pop())

min=math.pow(10,15)


for i in range(100000):
    #print (population)
    values=[]
    for j in range(population_size):
        values.append(calc_value(population[j]))
    temp_values=values[:]
    temp_values.sort(reverse=True)
    if np.amin(values) < min:
        min=np.amin(values)
    print('minimum in this generation is '+str(np.amin(values)))
    print('average fitness in '+str(i+1)+' th generation is '+str(np.mean(values)))
    mating_pool=[]
    for j in range(population_size):
        flag=-1
        for k in range(population_size):
            if(values[k]==temp_values[j]):
                flag=k
                break;
        for l in range(int(math.pow(1.05,j+1))):
            mating_pool.append(population[flag])
    
    
    new=0
    
    for j in range(int(population_size/2)):
        #Select ParentA
        a=randint(0,len(mating_pool)-1)
        b=randint(0,len(mating_pool)-1)
        parentA1=mating_pool[a]
        parentA2=mating_pool[b]
        #print(parentA1)
        #print(parentA2)
        #print(calc_value(parentA1))
        #print(calc_value(parentA2))
        
        if calc_value(parentA1) < calc_value(parentA2):
            parentA=parentA1
        else:
            parentA=parentA2
        #print(parentA)    
        #Select ParentB
        c=randint(0,len(mating_pool)-1)
        d=randint(0,len(mating_pool)-1)
        parentB1=mating_pool[c]
        parentB2=mating_pool[d]
        if calc_value(parentB1) < calc_value(parentB2):
            parentB=parentB1
        else:
            parentB=parentB2
        child1=[]
        child2=[]    
        crossover_rate=0.8
        if random.uniform(0,1)< crossover_rate:
            
            xD=random.uniform(0,1)
            #print(xD)
            
            for k in range(D):
                child1.append(xD*parentA[k] + (1-xD)*parentB[k])
            for k in range(D):
                child2.append((1-xD)*parentA[k] + (xD)*parentB[k])
        else:
            child1=parentA
            child2=parentB
        #print(child1)
        #print(child2)
        #MUTATION
        
       
        if i < 5000:
           mutation_rate=0.2
           for k in range(D):
                if random.uniform(0,1) < mutation_rate:
                    child1[k]=child1[k]+float(np.random.randn(1,1)/3)
           for k in range(D):
                if random.uniform(0,1) < mutation_rate:
                    child2[k]=child2[k]+float(np.random.randn(1,1)/3)
        else:
            mutation_rate=0.02
            
            for k in range(D):
                if random.uniform(0,1) < mutation_rate:
                    child1[k]=child1[k]+float(np.random.randn(1,1)/3)
            for k in range(D):
                if random.uniform(0,1) < mutation_rate:
                    child2[k]=child2[k]+float(np.random.randn(1,1)/3)
            
        #print(child1)
        #print(child2)
        population[new]=child1
        new=new+1
        population[new]=child2
        new=new+1
print('minimum value obtained is '+ str(min))