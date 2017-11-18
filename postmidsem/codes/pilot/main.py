import array
import random

import numpy
from math import sqrt

from deap import algorithms
from deap import base
from deap import benchmarks
from deap.benchmarks.tools import diversity, convergence
from deap import creator
from deap import tools

from population import *
from network import Neterr
from chromosome import Chromosome, crossover

n_hidden = 100
indim = 8
outdim = 1
creator.create("FitnessMin", base.Fitness, weights=(-1.0, -1.0, -1.0, -1.0))
creator.create("Individual", Chromosome, fitness=creator.FitnessMin)

toolbox = base.Toolbox()


def minimize(individual):
    network_obj = Neterr(indim, outdim, n_hidden, np.random)
    outputarr = network_obj.feedforward_ne(individual)

    neg_log_likelihood_val = give_neg_log_likelihood(outputarr, network_obj.resty)
    mean_square_error_val = give_mse(outputarr, network_obj.resty)
    false_positve_rat = give_false_positive_ratio(outputarr, network_obj.resty)
    false_negative_rat = give_false_negative_ratio(outputarr, network_obj.resty)

    return neg_log_likelihood_val, mean_square_error_val, false_positve_rat, false_negative_rat


def mycross(ind1, ind2, gen_no):
    child1 = crossover(ind1, ind2, gen_no, inputdim=8, outputdim=1)
    child2 = crossover(ind1, ind2, gen_no, inputdim=8, outputdim=1)

    return child1, child2


def mymutate(ind1):
    new_ind = ind1.do_mutation(0.2, 0.1, 0.05, indim, outdim, n_hidden, numpy.random)
    return new_ind


def initIndividual(ind_class, inputdim, outputdim):
    ind = ind_class(inputdim, outputdim)
    return ind


toolbox.register("individual", initIndividual, creator.Individual, indim, outdim)
toolbox.register("population", tools.initRepeat, list, toolbox.individual)

toolbox.register("evaluate", minimize)
toolbox.register("mate", mycross)
toolbox.register("mutate", mymutate)
toolbox.register("select", tools.selNSGA2)


def main(seed=None):
    random.seed(seed)

    NGEN = 100
    MU = 4 * 15  # this has to be a multiple of 4. period.
    CXPB = 0.9
    file_ob = open("log.txt", "w")
    file_ob.write(str('swapnil'))
    file_ob.close()
    stats = tools.Statistics(lambda ind: ind.fitness.values[1])
    # stats.register("avg", numpy.mean, axis=0)
    # stats.register("std", numpy.std, axis=0)
    stats.register("min", numpy.min, axis=0)
    stats.register("max", numpy.max, axis=0)

    logbook = tools.Logbook()
    logbook.header = "gen", "evals", "std", "min", "avg", "max"
    pop = toolbox.population(n=MU)

    # Evaluate the individuals with an invalid fitness
    invalid_ind = [ind for ind in pop if not ind.fitness.valid]
    fitnesses = toolbox.map(toolbox.evaluate, invalid_ind)
    for ind, fit in zip(invalid_ind, fitnesses):
        ind.fitness.values = fit

    # This is just to assign the crowding distance to the individuals
    # no actual selection is done
    pop = toolbox.select(pop, len(pop))
    # print(pop)
    record = stats.compile(pop)
    logbook.record(gen=0, evals=len(invalid_ind), **record)
    print(logbook.stream)
    maxi = 0
    stri = ''
    # Begin the generational process
    # print(pop.__dir__())
    for gen in range(1, NGEN):

        # Vary the population
        offspring = tools.selTournamentDCD(pop, len(pop))
        offspring = [toolbox.clone(ind) for ind in offspring]

        # print("changed?", gen)
        # print(maxi)
        # print("length",len(offspring))
        for ind1, ind2 in zip(offspring[::2], offspring[1::2]):
            # print(ind1.fitness.values)
            if random.random() <= CXPB:
                toolbox.mate(ind1, ind2, gen)
            maxi = max(maxi, ind1.node_ctr, ind2.node_ctr)
            toolbox.mutate(ind1)
            toolbox.mutate(ind2)
            del ind1.fitness.values, ind2.fitness.values

        # Evaluate the individuals with an invalid fitness
        invalid_ind = [ind for ind in offspring if not ind.fitness.valid]
        fitnesses = toolbox.map(toolbox.evaluate, invalid_ind)
        for ind, fit in zip(invalid_ind, fitnesses):
            ind.fitness.values = fit

        # Select the next generation population
        pop = toolbox.select(pop + offspring, MU)

        record = stats.compile(pop)
        logbook.record(gen=gen, evals=len(invalid_ind), **record)
        anost = logbook.stream
        print(anost)
        stri += anost + '\n'
        # file_ob.write(str(logbook.stream))
        # print(len(pop))
        # file_ob.close()
    print(stri)
    file_ob = open("log.txt", "w")
    file_ob.write(stri)
    file_ob.close()
    return pop, logbook


if __name__ == "__main__":
    pop, stats = main()
    import random
    fronts = tools.sortNondominated(pop, len(pop))
    if len(fronts[0]) < 30:
        pareto_front = fronts[0]
    else:
        pareto_front =  random.choice(fronts[0], 30)
    print("Pareto Front: ",[ item.fitness.values for item in pareto_front])
    neter = Neterr(indim, outdim, n_hidden, np.random)
    file_ob = open("log.txt", "a")
    print("\ntest on one with min validation error", neter.test_err(min(pop, key=lambda x: x.fitness.values[1])))
    file_ob.write(
        "test on one with min validation error " + str(neter.test_err(min(pop, key=lambda x: x.fitness.values[1]))))
    file_ob.close()
    # file_ob.write( "test on one with min validation error " + str(neter.test_err(min(pop, key=lambda x: x.fitness.values[1]))))

    # print(stats)
    '''
    import matplotlib.pyplot as plt
    import numpy
    
    front = numpy.array([ind.fitness.values for ind in pop])
    plt.scatter(front[:,0], front[:,1], c="b")
    plt.axis("tight")
    plt.show()'''
