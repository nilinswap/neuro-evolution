from random import randint
import math
population=[]

def random_str():
    s=""
    for i in range(5):
        s=s+str(randint(0,1))
    return s    

def calculate_value(s):
    return (16*int(population[s][0]) + 8*int(population[s][1]) + 4*int(population[s][2]) + 2*int(population[s][3]) + 1*int(population[s][4])) 

# Create Population
for i in range(4):
    population.append(random_str())



for i in range(200):
    print()
    print()
    #print(population)
    values=[]
    s=0;
    for j in range(4):
        values.append(calculate_value(s))
        s=s+1
    #print(values)
    # CALCULATE AVERAGE FITNESS
    avg_fitness=0
    for j in range(4):
        avg_fitness=avg_fitness + values[j]
    avg_fitness=avg_fitness/4
    print("avg fitness in "+str(i+1)+"th population is "+str(avg_fitness))
    
    
    
    #CALCULATE MAXIMUM FITNESS
    max_fitness=-1
    for j in range(4):
        if values[j] > max_fitness:
            max_fitness=values[j]
    print("maximum fitness in "+str(i+1)+"th population is "+str(max_fitness))
    
    
    
    flag=1
    for j in range(4):
        if values[j]==31:
            print("Target string found in "+ str(i+1) + "th "+ "population")
            flag=2
            break
    if flag==2:
        break
    fitness=[]
    for j in range(4):
        fitness.append(int(math.floor(values[j]*values[j]/10)))
    #print(fitness)
    matingpool=[]
    for j in range(4):
        for k in range(fitness[j]):
            matingpool.append(population[j])
   #print(matingpool)
    sum_fitness=0;
    for j in range(4):
        sum_fitness+=fitness[j]
    #print(sum_fitness)
    new=0
    for j in range(2):
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
        cross_site=randint(1,4)
        #print(cross_site)
        for k in range(cross_site,5):
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
        "".join(parentA)
        "".join(parentB)
        population[new]=parentA
        new=new+1
        population[new]=parentB
        new=new+1
    

    