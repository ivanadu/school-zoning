import cvxpy as cp
import numpy as np
import pickle

np.random.seed(1)

class SchoolAssigner:
    """
    Assign kindergarten students to schools using the modified optimization
    problem described in the problem set.
    """

    def __init__(self, path):
        """Initializes instance of class using the optimization inputs in path
        path = file path"""

        #load input data
        pickle_in = open(path, 'rb')
        inputs = pickle.load(pickle_in)

        #assign inputs to attributes
        self.a = inputs['a']
        self.q = inputs['q']
        self.y = inputs['y']
        self.D = inputs['D']
        self.dist_range = inputs['dist_range']
        self.div_range = inputs['div_range']

        #derive some more convenient forms
        self.A = np.diag(self.a)
        self.Y = np.diag(self.y)

        #dims
        self.m = len(self.q)
        self.n = len(self.a)

    def assign_students(self, l1):
        """Runs assignment of students to schools and compute distance and
        diversity objectives
        l1 = importance of distance objective vs diversity objective"""

        np.random.seed(1)

        #normalize l1 if < 0 or > 1
        l1 = np.min((l1, 1))
        l1 = np.max((l1, 0))
        l1 = np.round(l1, 2)

        print('Optimizing assignment for lambda = %s...'%l1)

        #optimization variable
        X = cp.Variable((self.m, self.n))

        #individual objectives
        dist_obj = (1/np.sum(self.a))*cp.trace(X@self.A@self.D)
        div_obj = (0.5*cp.sum(cp.abs(X@self.Y@self.a -
        (self.Y@self.a@np.ones(self.n))/(self.q@np.ones(
        self.m)) * self.q))) / np.sum(self.Y@self.a)

        #combined objective
        if l1 == 1:
            combined_obj = dist_obj
        else:
            combined_obj = l1 * (dist_obj)  + (1-l1) * (div_obj * 0.5)

        #constraints
        constraints = [
        cp.sum(X, axis = 0) <= np.ones(self.n), #proportions are relative to type
        X >= np.zeros((self.m, self.n)), #assignment is a proportion gte 0
        X <= np.ones((self.m, self.n)), #assignment is a proportion lte 1
        X@self.a <= self.q, #do not exceed school capacity
        X@self.a@np.ones(self.m) == np.sum(self.a) #assign all students
        ]

        #solve problem
        prob = cp.Problem(cp.Minimize(combined_obj), constraints)
        prob.solve(solver = 'OSQP')

        if prob.status == 'optimal':
            print('Solution converged in %.2f seconds'%prob.solver_stats.solve_time)
        else:
            print('Solution did not converge.')

        #save assignment values
        self.distance = self.calc_distance(X.value)
        self.diversity = self.calc_diversity(X.value)
        self.lam = l1

    def calc_distance(self, X):
        """Calculates distance objective given an assignment matrix X"""
        dist = (1/np.sum(self.a))*np.trace(X@self.A@self.D)
        if dist > 1.5:
            dist = 1.5
        return dist

    def calc_diversity(self, X):
        """Calculates diversity objective given an assignment matrix X"""
        return (0.5*np.sum(np.abs(X@self.Y@self.a -
        (self.Y@self.a@np.ones(self.n))/(self.q@np.ones(
        self.m)) * self.q))) / np.sum(self.Y@self.a)
