from random import randint
import math
import random
population=[]
ls=13
def random_str():
    s=""
    for i in range(ls):
        s=s+str(randint(0,1))
    return s    

def calculate_value(s):
    sum=0
    j=0
    for i in range(ls-1,-1,-1):
        sum+=math.pow(2,i)*int(population[s][j])
        j=j+1
    return sum
# Create Population
for i in range(100):
    population.append(random_str())



for i in range(200):
    print()
    print()
    #print(population)
    values=[]
    s=0;
    for j in range(100):
        values.append(calculate_value(s))
        s=s+1
    #print(values)
    # CALCULATE AVERAGE FITNESS
    
    avg_fitness=0
    for j in range(100):
        avg_fitness=avg_fitness + values[j]
    avg_fitness=avg_fitness/100
    print("avg fitness in "+str(i+1)+"th population is "+str(avg_fitness))
    
    
    
    #CALCULATE MAXIMUM FITNESS
    max_fitness=-1
    for j in range(100):
        if values[j] > max_fitness:
            max_fitness=values[j]
    print("maximum fitness in "+str(i+1)+"th population is "+str(max_fitness))
    
    
    
    flag=1
    for j in range(100):
        if values[j]==8191:
            print("Target string found in "+ str(i+1) + "th "+ "population")
            flag=2
            break
    if flag==2:
        break
    fitness=[]
    for j in range(100):
        fitness.append(int(math.floor(values[j]/10)))
    #print(fitness)
    matingpool=[]
    for j in range(100):
        for k in range(fitness[j]):
            matingpool.append(population[j])
    #print(matingpool)
    sum_fitness=0;
    for j in range(100):
        sum_fitness+=fitness[j]
    #print(sum_fitness)
    new=0
    for j in range(50):
        a=randint(0,sum_fitness)
        b=randint(0,sum_fitness)
     #   print(a)
      #  print(b)
        parentA=matingpool[a-2]
        
        parentB=matingpool[b-2]
        #print(parentA)
        #print(parentB)
        parentA=list(parentA)
        parentB=list(parentB)
       # print("HI")
        cross_site=randint(1,ls-1)
        #print(cross_site)
        mutation_rate=0.15
        for k in range(cross_site,ls):
            if parentA[k]=='0':
                parentA[k]='1'
            else:
                parentA[k]='0'
            if parentB[k]=='0':
                parentB[k]='1'
            else:
                parentB[k]='0'
        
        # Introduce Mutation
        if parentA==parentB:
            if parentA[k]=='0':
                parentA[k]='1'
            else:
                parentA[k]='0'
        else:
            for l in range(ls):
                x=random.uniform(0,1)
                if x < mutation_rate:
                    if parentA[l]=='0':
                        parentA[l]='1'
                    else:
                        parentA[l]='0'
            for z in range(ls):
                y=random.uniform(0,1)
                if y < mutation_rate:
                    if parentA[z]=='0':
                        parentA[z]='1'
                    else:
                        parentA[z]='0'
                    
        "".join(parentA)
        "".join(parentB)
        population[new]=parentA
        new=new+1
        population[new]=parentB
        new=new+1
    #print('('+str(i+1)+","+str(avg_fitness)+')')
    

    