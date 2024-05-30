import numpy as np

class SOP_Node:
    def __init__(self, time,I=1, A1=0, A2=0, p1=0.9, pd1=0.1, pd2=0.02, Lplus=0.1, Lminus=0.01, learning = 0.15):
        self.I = np.zeros(time)
        self.I[0] = 1
        self.A1 = np.zeros(time)
        self.A2 = np.zeros(time)
        
        self.p1 = p1
        self.pd1 = pd1
        self.pd2 = pd2
        
        self.V = {}
        self.Lplus = Lplus
        self.Lminus = Lminus
        self.learning = learning
        
        self.p2 = np.zeros(time)

    def associate(self, time,nodes):
        if not isinstance(nodes, list):
            nodes = [nodes]
        
        if not all(isinstance(node,SOP_Node) for node in nodes):
            raise TypeError("All arguments must be Nodes")

        for node in nodes:
            self.V[node] = np.zeros(time)

    def update(self,moment,stimulus, p1 = 0):
        if p1 != 0:
            self.p1 = p1
  
        dI = self.A2[moment]*self.pd2 - self.I[moment]*self.p1*stimulus - self.I[moment]*self.p2[moment]
        self.I[moment+1] = self.I[moment] + dI
        
        dA1 = self.I[moment]*self.p1*stimulus - self.A1[moment]*self.pd1
        self.A1[moment+1] = self.A1[moment] + dA1

        dA2 = self.A1[moment]*self.pd1 + self.I[moment]*self.p2[moment] - self.A2[moment]*self.pd2
        self.A2[moment+1] = self.A2[moment] + dA2

        #Check if there is any association
        if self.V:
            association_sum = 0
        
            for node in self.V:
                dV = node.A1[moment]*self.learning*(self.A1[moment]*self.Lplus - self.A2[moment]*self.Lminus)
                self.V[node][moment+1] = self.V[node][moment] + dV
                association_sum += node.A1[moment+1]*self.V[node][moment+1]
            
            self.p2[moment+1] = max(min(association_sum,1),0)


class MTS:
    def __init__(self, time, n_integrators=2, X1=0.9, theta=0, lambda_a=3.9, lambda_b=1.3):
        self.n_integrators = n_integrators
        self.theta = theta
        
        self.a = np.array([[1 - np.exp(-lambda_a * (i + 1))] for i in range(n_integrators)])
        self.b = np.array([[np.exp(-lambda_b * (i + 1))] for i in range(n_integrators)])

        self.X = np.zeros((n_integrators+1, time))
        self.X1 = X1
        
        self.V = np.zeros((n_integrators, time))
        
    def update(self, version, moment, stimulus, X1 = 0):
        if X1 != 0:
            self.X1 = X1
        
        self.X[0, moment] = stimulus*self.X1

        for i in range(self.n_integrators):
            diff_XiVi = self.X[i,moment] - self.V[i,moment]
            self.X[i+1,moment] = np.where(diff_XiVi > self.theta, diff_XiVi, self.theta)

        if version == "FB":
            dV = self.b*self.X[1:, moment:moment+1] - (1 - self.a)*self.V[:, moment:moment+1]

        elif version == "FF":
            dV = self.b*self.X[0:-1, moment:moment+1] - (1 - self.a)*self.V[:, moment:moment+1]

        self.V[:, moment+1:moment+2] = self.V[:, moment:moment+1] + dV

def trials(n, period, delay):
    return [int(delay + i*period) for i in range(n)]

def test(test_after, last_trial):
    if not isinstance(test_after,list):
        test_after = [test_after]

    return [int(last_trial + 1 + t) for t in test_after]
