# -*- coding: utf-8 -*-
import numpy as np
import geatpy as ea  # 导入geatpy库


class moea_NSGA2_templet(ea.MoeaAlgorithm):
    """
        moea_NSGA2_templet : class - Multi-objective evolutionary NSGA-II algorithm class

        Algorithm description:
        NSGA-II is used for multi-objective optimization. For details of the algorithm, please refer to reference [1].

        References:
        [1] Deb K , Pratap A , Agarwal S , et al. A fast and elitist multiobjective
        genetic algorithm: NSGA-II[J]. IEEE Transactions on Evolutionary
        Computation, 2002, 6(2):0-197.

    """

    def __init__(self,
                 problem,
                 population,
                 MAXGEN=None,
                 MAXTIME=None,
                 MAXEVALS=None,
                 MAXSIZE=None,
                 logTras=None,
                 verbose=None,
                 outFunc=None,
                 drawing=None,
                 dirName=None,
                 args=None,
                 **kwargs):
        
        super().__init__(problem, population, MAXGEN, MAXTIME, MAXEVALS, MAXSIZE, logTras, verbose, outFunc, drawing,
                         dirName)
        if population.ChromNum != 1:
            raise RuntimeError('The population object passed in must be a single-chromosome population type.')
        self.name = 'NSGA2'
        self.args = args
        if self.problem.M < 10:
            self.ndSort = ea.ndsortESS  
        else:
            self.ndSort = ea.ndsortTNS  
        self.selFunc = 'tour' 
        if population.Encoding == 'P':
            self.recOper = ea.Xovpmx(XOVR=1)  
            self.mutOper = ea.Mutinv(Pm=1) 
        elif population.Encoding == 'BG':
            self.recOper = ea.Xovud(XOVR=1)  
            self.mutOper = ea.Mutbin(Pm=None) 
        elif population.Encoding == 'RI':
            self.recOper = ea.Recsbx(XOVR=1, n=20)  
            self.mutOper = ea.Mutpolyn(Pm=1 / self.problem.Dim, DisI=20)  
        else:
            raise RuntimeError('The encoding must be ''BG'', ''RI'' or ''P''.')

    def reinsertion(self, population, offspring, NUM):

        """
        Description:
            Reinsert individuals to generate a new generation of population (using the strategy of parent-child merging selection).
            NUM is the number of individuals that need to be retained to the next generation.
            Note: Here, an equivalent modification is made to the original NSGA-II: first calculate the fitness of the individuals in the population according to the Pareto classification and crowding distance,
            then call the dup selection operator (see help(ea.dup) for details) to select individuals to be retained in the next generation in descending order of fitness.
            This is exactly the same as the result obtained by the selection method of the original NSGA-II.

        """


        population = population + offspring

        [levels, _] = self.ndSort(population.ObjV, NUM, None, population.CV, self.problem.maxormins)  
        dis = ea.crowdis(population.ObjV, levels)  
        population.FitnV[:, 0] = np.argsort(np.lexsort(np.array([dis, -levels])), kind='mergesort') 
        chooseFlag = ea.selecting('dup', population.FitnV, NUM) 
        return population[chooseFlag]

    def call_aimFunc(self, pop, call_real=False):

        """
        Description: Call aimFunc() or evalVars() of the problem class to complete the calculation of the population objective function value and the degree of constraint violation.

        For example: if population is a population object, call call_aimFunc(population) to complete the calculation of the objective function value.
            After that, the objective function value can be obtained through population.ObjV, and the constraint violation degree matrix can be obtained through population.CV.

        Input parameters:
            pop : class <Population> - population object.

        Output parameters:
            No output parameters.       

        """

        pop.Phen = pop.decoding()  
        if self.problem is None:
            raise RuntimeError('error: problem has not been initialized.')
        self.problem.evaluation(pop, call_real)  
        self.evalsNum = self.evalsNum + pop.sizes if self.evalsNum is not None else pop.sizes  

        if not isinstance(pop.ObjV, np.ndarray) or pop.ObjV.ndim != 2 or pop.ObjV.shape[0] != pop.sizes or \
                pop.ObjV.shape[1] != self.problem.M:
            raise RuntimeError('error: ObjV is illegal.')
        if pop.CV is not None:
            if not isinstance(pop.CV, np.ndarray) or pop.CV.ndim != 2 or pop.CV.shape[0] != pop.sizes:
                raise RuntimeError('error: CV is illegal. ')

    def run(self, prophetPop=None): 
        # =====================================================
        population = self.population
        NIND = population.sizes
        self.initialization()  
        # =======================================================
        population.initChrom()  
        
        if prophetPop is not None:
            population = (prophetPop + population)[:NIND]  
        self.call_aimFunc(population)
        [levels, _] = self.ndSort(population.ObjV, NIND, None, population.CV, self.problem.maxormins)  
        population.FitnV = (1 / levels).reshape(-1, 1)  
        # =======================================================
        while not self.terminated(population):

            offspring = population[ea.selecting(self.selFunc, population.FitnV, NIND)]
            offspring.Chrom = self.recOper.do(offspring.Chrom)
            # if self.currentGen > self.MAXGEN * 0.5:
            #     offspring.Chrom = ea.mutmani(offspring.Encoding, offspring.Chrom, offspring.Field, self.problem.M-1)
            offspring.Chrom = self.mutOper.do(offspring.Encoding, offspring.Chrom, offspring.Field) 
            self.call_aimFunc(offspring)  
            population = self.reinsertion(population, offspring, NIND)  
        return self.finishing(population) 
