import random
import copy 
import itertools
import numpy as np
from tqdm import tqdm

# uncomment and install these if necessary
# from log import Log
# from sklearn.utils import shuffle as sk_shuffle
# from concurrent.futures import ProcessPoolExecutor
# from concurrent.futures import as_completed
# from nsga2 import nsga_2_pass,dominates

def ga_concat(x):
    return list(itertools.chain.from_iterable(x))

def es_sigma(_min, _max):
    return  (_max-_min)  / 2.0 / 3.1 

def es_gaussian_noise(_min, _max):
    mean = (_max+_min)  / 2.0 
    dev = es_sigma(_min,_max)
    ra = random.gauss(mean, dev)
    while (_min <= ra <= _max) == False:
        ra = random.gauss(mean, dev)
    return ra

class Individual(object):

    def __init__(self, 
                 Nf,
                 filters, 
                 fitness_max):
        self.genotype = [random.randrange(0, len(filters)) for _ in range(Nf)]
        self.params = []
        self.filters = filters
        for fid in self.genotype:
            self.params += [d.value for d in self.filters[fid].domains]
        self.fitness_max = fitness_max
        self.fitness = fitness_max
    
    def apply(self, image, params = None):
        if params is None:
            params = self.params
        ilast = 0
        for fid in self.genotype:
            ifilter = self.filters[fid]
            image = ifilter(image,*params[ilast:ilast+ifilter.nparams()])
            ilast += ifilter.nparams()
        return image

    def change(self, i, j, rand_params = False):
        p_i = 0
        for p in range(i):
            p_i += len(self.filters[self.genotype[p]].domains)
        e_i = p_i + len(self.filters[self.genotype[i]].domains)
        if rand_params == False:
            self.params = self.params[:p_i] + [d.value for d in self.filters[j].domains] + self.params[e_i:]
        else:
            self.params = self.params[:p_i] + [d.random() for d in self.filters[j].domains] + self.params[e_i:]
        self.genotype[i] = j 

    def pice(self, s=0, e=None):
        if e is None:
            e = len(self.genotype)
        new = copy.copy(self)
        new.fitness = new.fitness_max
        new.genotype = self.genotype[s:e]
        p_s = 0
        for i in range(s):
            p_s += len(self.filters[self.genotype[i]].domains)
        p_e = p_s
        for i in range(s,e):
            p_e += len(self.filters[self.genotype[i]].domains)
        new.params = self.params[p_s:p_e]
        return new

    def __add__(self, other):
        new = copy.copy(self)
        new.fitness = new.fitness_max
        new.genotype += other.genotype
        new.params += other.params
        return new
    
    def __len__(self):
        return len(self.genotype)

    def nparams(self):
        return len(self.params)

class ParamerESOptimizer(object):

    def __init__(self, fitness):  
        self.fitness = fitness      
        self.init_learning_rate = 0.1
        self.decay = 0.75
        self.popsize = 5
        self.ngens = 3
        self.learning_rate = self.init_learning_rate
    
    def _start_individual_from_domain(self):
        genotype = []
        for i, d in enumerate(self.domains):
            genotype.append(d.value)
        return genotype

    def _gen_offspring(self, n):
        S = []
        P = [[0 for _ in range(n.nparams())] for _ in range(self.popsize)]
        for i in range(self.popsize):
            j = 0
            for f in n.genotype:
                for d in n.filters[f].domains:
                    sigma = es_sigma(*d.domain) * 0.25
                    nparam = n.params[j] + random.gauss(0, sigma)
                    nparam = max(d.domain[0], min(nparam, d.domain[1]))
                    P[i][j] = d.ptype.ptype(nparam)
                    S.append(sigma)
                    j += 1
        return P,S

    @staticmethod
    def _normalize(n):
        j = 0
        for f in n.genotype:
            for d in n.filters[f].domains:
                n.params[j] = max(d.domain[0], min(n.params[j], d.domain[1]))
                n.params[j] = d.ptype.ptype(n.params[j])
                j += 1

    def _evaluate(self, n, X, Y, P):
        R = np.zeros(len(P))
        for p in range(len(P)):
            _Xf = np.array([n.apply(X[i], P[p]) for i in range(X.shape[0])])
            R[p] = self.fitness(_Xf, X, Y)
            del _Xf
        return R

    def _compute_gradient(self, R):
        std = np.std(R)
        if std == 0.0:
            return np.zeros_like(R)
        return (R - np.mean(R)) / std

    def step(self, n, X, Y):
        P,S = self._gen_offspring(n)
        R = self._evaluate(n, X, Y, P)
        A = self._compute_gradient(R)
        L = n.nparams()
        P = np.array(P)
        grad = np.dot(P.T,A)
        if len(grad.shape) and grad.shape[0]:
            for i in range(L):
                n.params[i] += (self.learning_rate / (L*S[i])) * grad[0]
        self.learning_rate *= self.decay
        #delate all np arrays
        del P, S, R, A, L

    def fit(self, n, X, Y):
        self.learning_rate = self.init_learning_rate
        for _ in range(self.ngens):
            self.step(n, X, Y)
        self._normalize(n)
        return n

class PatamerGAOptimizer(object):

    def __init__(self, fitness, compare):  
        self.fitness = fitness     
        self.compare = compare 
        self.popsize = 5
        self.ngens = 3
        self.population = []
        self.newpopulation = []
        self.domains = []
        self.elite = []

    def one_point_crossover(self, x, y):
        index = random.randrange(1,len(x)-1)
        return x[0:index] + y[index:]

    def eval_params(self, n, X, Y, P):
        Xf = np.array([n.apply(X[i], P) for i in range(X.shape[0])])
        return self.fitness(Xf, X, Y)

    def crossover(self):
        if len(self.domains) > 1:
            p = self.population[random.randrange(0, len(self.population))]
            m = self.population[random.randrange(0, len(self.population))]
            y = self.one_point_crossover(p,m)
        else:
            y = self.population[random.randrange(0, len(self.population))]
        return y

    def mutation(self, x):
        for i, d in enumerate(self.domains):
            if random.uniform(0,1) < 0.5:
                x[i] = d.random()
        return x

    def selection(self, n, X, Y, m, i):
        if  self.compare(self.eval_params(n, X, Y, m), self.eval_params(n, X, Y, self.population[i])):
            self.newpopulation[i] = m
        else:
            self.newpopulation[i] = self.population[i]

    def elitism(self, n, X, Y, m):
        if self.compare(self.eval_params(n, X, Y, m), self.eval_params(n, X, Y, self.elite)):
            self.elite = m

    def step(self, n, X, Y, i):
        y = self.crossover()
        m = self.mutation(y)
        self.selection(n, X, Y, m, i)
        self.elitism(n, X, Y, m)

    def fit(self, n, X, Y):
        self.domains = ga_concat([[d for d in n.filters[fid].domains] for fid in n.genotype])
        self.population = [
            ga_concat([[d.random() for d in n.filters[fid].domains] for fid in n.genotype]) for p in range(self.popsize)
        ]
        self.newpopulation = [None for p in range(self.popsize)]
        self.elite = n.params
        for _ in range(self.ngens):
            for i in range(self.popsize):
                self.step(n, X, Y, i)
            self.population = self.newpopulation
            self.newpopulation = [None for p in range(self.popsize)]
        n.params = self.elite
        return n

class AGVOptimizer(object):

    @staticmethod
    def mutation(x, domain):
        for i in range(len(x)):
            if random.uniform(0,1) < 0.5:
                x.change(i, random.randint(domain[0],domain[1]), rand_params=True)
        return x

    @staticmethod
    def mutation_random_params(x):
        j = 0
        for f in x.genotype:
            for d in x.filters[f].domains:
                if random.uniform(0,1) < 0.5:
                    x.params[j] = d.random()
                j += 1
        return x

    @staticmethod
    def one_point_crossover(x,y):
        index = random.randrange(1,len(x)-1)
        return x.pice(0,index) + y.pice(index)

    @staticmethod
    def two_point_crossover(x,y):
        index_s = random.randrange(1,len(x)-1)
        index_e = random.randrange(index_s,len(x))
        return x.pice(0,index_s) + y.pice(index_s,index_e) + x.pice(index_e)

    def selection_raking(self, offsprings):
        #select
        all_elements = self.population + offsprings
        all_elements = sorted(all_elements, key=lambda x: x.fitness)
        self.population = all_elements[0:len(self.population)]

    def selection_pareto(self, offsprings):
        #select
        all_elements = [offspring for offspring in offsprings]
        all_elements+= [parent for parent in self.population]
        new_pop = nsga_2_pass(len(self.population), [e.fitness for e in all_elements])
        self.population = [all_elements[p] for p in new_pop]

    def __init__(self, 
                 Nf,   
                 filters,
                 fitness,
                 NP,
                 fitness_max = float("inf"), 
                 params_strategy = "direct", # or tournament
                 params_optimizer = "ES", # GA or random
                 params_pool = "offsprings", # and/or "|parents"
                 selection_type = "ranking", # or pareto (|no-params)
                 use_elitims = True
                 ):
        self.population = [Individual(Nf,
                                      filters,  
                                      fitness_max) for _ in range(NP)]
        self.filters = filters
        self.fitness = fitness
        self.ga_domain = [0, len(filters)-1]
        self.params_strategy = params_strategy
        self.params_optimizer = params_optimizer
        self.params_pool = params_pool
        self.use_elitims = use_elitims
        self._pbest = None
        self._first= True
        #test
        if  selection_type.find("pareto") >= 0:
            sq_dis_to_0 = lambda x: (x[0]+x[1])**2 
            self.compare_tournament = lambda f1, f2 : dominates(f1, f2)
            self.compare_elit = lambda f1, f2 : sq_dis_to_0(f1) <=  sq_dis_to_0(f2)
            if selection_type.find("no-params") >= 0:
                #for params
                self.compare_params = lambda f1, f2 : f1 <= f2 
                self.fitness_params = lambda Xf, X, Y: self.fitness(Xf,X,Y, True)
                self.selection_type = "pareto"
            else:
                #for params
                self.compare_params = lambda f1, f2 : sq_dis_to_0(f1) <=  sq_dis_to_0(f2)
                self.fitness_params = self.fitness
                if params_optimizer == "ES":            
                    self.fitness_params = lambda Xf, X, Y: sq_dis_to_0(self.fitness(Xf,X,Y))
            #set as pareto
            self.selection_type = "pareto"
        else:
            self.compare_tournament = \
            self.compare_params = \
            self.compare_elit = lambda f1, f2 : f1 <= f2 
            #for params
            self.compare_params = lambda f1, f2 : f1 <= f2 
            self.fitness_params = self.fitness
            #set as ranking
            self.selection_type = "ranking"

    def evaluate(self, n, X, Y):
        Xf = np.array([n.apply(X[i]) for i in range(X.shape[0])])
        return self.fitness( Xf, X, Y )
    
    def evaluate_set(self, set_to_eval, X, Y):
        for n in set_to_eval:
            n.fitness = self.evaluate(n, X, Y)

    def evaluate_population(self, X, Y):
        self.evaluate_set(self.population, X, Y)

    def optimize_params_and_eval(self, n, X, Y):
        if self.params_optimizer == "ES":
            ParamerESOptimizer(self.fitness_params).fit(n, X, Y)
            n.fitness = self.evaluate(n, X, Y)
            return n
        if self.params_optimizer == "GA":
            PatamerGAOptimizer(self.fitness_params, self.compare_params).fit(n, X, Y)
            n.fitness = self.evaluate(n, X, Y)
            return n
        elif self.params_optimizer == "random":
            AGVOptimizer.mutation_random_params(n)
            n.fitness = self.evaluate(n, X, Y)
            return n
        else:
            n.fitness = self.evaluate(n, X, Y)
            return n

    def apply_params_strategy(self, offsprings, X, Y):
        if self.params_strategy == "direct":
            for i,n in enumerate(offsprings):
                offsprings[i] = self.optimize_params_and_eval(n, X, Y)
        elif self.params_strategy == "tournament":
            for i,n in enumerate(offsprings):
                n.fitness = self.evaluate(n, X, Y)
                n1 = self.optimize_params_and_eval(copy.deepcopy(n), X, Y)
                if self.compare_tournament(n1.fitness, n.fitness):
                    offsprings[i] = n1
                else:
                    offsprings[i] = n
        else: #None
            pass
        return offsprings

    def elitism(self, X, Y):
        if self._pbest is None:
            self._pbest = self.population[0]
        else:
            self.evaluate(self._pbest, X, Y)
        for p in self.population:
            if self.compare_elit(p.fitness,self._pbest.fitness):
                self._pbest = p

    def gen_offspring(self):
        # TODO: per i parametri dei genitori non eseguire ES
        p = self.population[random.randrange(0, len(self.population))]
        m = self.population[random.randrange(0, len(self.population))]
        y = self.one_point_crossover(p, m)
        n = self.mutation(y, self.ga_domain)
        return n

    def fit_pass(self, X, Y):
        #eval first pop
        if self._first:
            self.evaluate_set(self.population, X, Y)
            self._first = False
        #start
        offsprings = []
        for i in range(len(self.population)):
            offsprings.append(self.gen_offspring())
        #optimize params
        if self.params_pool.find("parent") >= 0: #or "parents"
            self.population = self.apply_params_strategy(self.population, X, Y)
        if self.params_pool.find("offspring") >= 0: #or "offspring"
            offsprings = self.apply_params_strategy(offsprings, X, Y)
        else:
            self.evaluate_set(offsprings, X, Y)
        #select
        if self.selection_type == "pareto":
            self.selection_pareto(offsprings)
        else:
            self.selection_raking(offsprings)
        #select the best
        if self.use_elitims:
            self.elitism(X, Y)
    
    def fit(self, X, Y, batch, epoch = 5, logs_path = "stats.txt"):
        self._first = True
        logs = Log(logs_path)
        for e in range(epoch):
            X, Y = sk_shuffle(X,Y)
            for i in tqdm(range(int(X.shape[0] / batch)), desc = "epoch {}/{} ".format(e+1,epoch)):
                s_i = i * batch
                e_i = (i+1) * batch
                batch_X, batch_Y = X[s_i:e_i],Y[s_i:e_i]
                self.fit_pass(batch_X, batch_Y)
                logs.log("{}\t{}\t".format(e,i) + "\t".join(["{},{},{}".format(str(p.genotype),str(p.params),str(p.fitness)) for p in self.population]))
                #free db
                del batch_X, batch_Y
        #return best
        if self.use_elitims:
            return self._pbest
        else:
            if self.selection_type == "pareto":
                return self.population[int(len(self.population)/2)]
            else:
                return self.population[0]
