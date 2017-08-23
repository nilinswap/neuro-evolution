# -*- coding: utf-8 -*-
"""
Created on Mon Aug 21 23:34:18 2017

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
    ans1=0
    ans2=0
    for i in range(D):
        ans1=ans1 + x[i]*x[i]
    for i in range(D):
        ans2=ans2+math.cos(2*math.pi*x[i])
    ans=-20.0*math.exp(-0.2*math.sqrt(ans1/D)) - math.exp(ans2/D) + 20 + math.e
    return ans    
population_size=(D+1)*10
for i in range(population_size):
    population.append(random_pop())

min=math.pow(10,15)

for i in range(200000):
    #print(population)
    print()
    print()
    values=[]
    fitness=[]
    for j in range(population_size):
        values.append(calc_value(population[j]))
    #print(values)
    if np.amin(values) < min:
        min=np.amin(values)
    print('minimum in this generation is '+str(np.amin(values)))
    #print(min)
    for j in range(population_size):
        #fitness.append(int(80/math.sqrt(values[j])))
        #fitness.append(int((math.pow(10,4)/(math.pow(values[j],1)))))
        fitness.append(int(1000/math.pow(1.2,values[j])))
    print('average fitness in '+str(i+1)+' th generation is '+str(np.mean(fitness)))
    #print()
    #print(fitness)
    
    mating_pool=[]
    for j in range(population_size):
        for k in range(fitness[j]):
            mating_pool.append(population[j])
    #print(mating_pool)
    #print(len(mating_pool))
    
     # Crossover
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
        
        xD=random.uniform(0,1)
        #print(xD)
        child1=[]
        child2=[]
        for k in range(D):
            child1.append(xD*parentA[k] + (1-xD)*parentB[k])
        for k in range(D):
            child2.append((1-xD)*parentA[k] + (xD)*parentB[k])
        
        #print(child1)
        #print(child2)
        #MUTATION
        
       
        if i < 5000:
            if D<5:
                 mutation_rate=0.05
            else:
                mutation_rate=0.03
            for k in range(D):
                if random.uniform(0,1) < mutation_rate:
                    child1[k]=child1[k]+float(np.random.randn(1,1))
            for k in range(D):
                if random.uniform(0,1) < mutation_rate:
                    child2[k]=child2[k]+float(np.random.randn(1,1))
        else:
            if D<5:
                 mutation_rate=0.03
            else:
                mutation_rate=0.02
            
            for k in range(D):
                if random.uniform(0,1) < mutation_rate:
                    child1[k]=child1[k]+float(np.random.randn(1,1)/30000)
            for k in range(D):
                if random.uniform(0,1) < mutation_rate:
                    child2[k]=child2[k]+float(np.random.randn(1,1)/30000)
            
        #print(child1)
        #print(child2)
        population[new]=child1
        new=new+1
        population[new]=child2
        new=new+1
print('minimum value obtained is '+ str(min))