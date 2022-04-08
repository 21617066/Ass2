import numpy as np

class NaiveB:
    def __init__(self):
        pass
    
    # Find class prob, class mean and class variance matrices
    def fit(self, data, lables):
        n = data.shape[0]
        self.classes = np.unique(lables)
        jmax = self.classes.shape[0]
        self.lables = lables
        self.pc = [0] * jmax
        self.u = [0] * jmax
        self.var = [0] * jmax
        
        for c in self.classes:
            x = data[np.flatnonzero(lables == c)]
            nc = x.shape[0]
            # Probability of class c
            self.pc[c] = nc/n
            
            # Mean of class c
            self.u[c] = np.sum(x, axis = 0)/nc
            
            # Assumed diagonal covariance matrix
            self.var[c] = np.var(x, axis = 0)
        
        # Make numpy arrays
        self.pc = np.array(self.pc)
        self.u = np.array(self.u)
        self.var = np.array(self.var)
        self.pxc = np.empty((n, jmax))
        
    def predict(self, data):
        
        # Probability of data point given class c
        for c in self.classes:
            dim_prob = (1/(np.sqrt(2*np.pi*self.var[c])))*np.exp(-0.5*(np.square(data-self.u[c]))/self.var[c])
            # print("dim prob:" + str(dim_prob))
            self.pxc[:,c] = np.prod(dim_prob, axis=1)
        
        print(self.pc.shape)
        self.prediction = np.argmax(self.pxc, axis=1)

            
            

    