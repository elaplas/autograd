from var import Var
import random
class Perceptron:
    def __init__(self, ninputs:int):
        self.size = ninputs
        self.weights = [Var(random.uniform(-1, 1), f"w {i}" ) for i in range(self.size)]
        self.bias = Var(random.uniform(-1, 1), "b")
    
    def __call__(self, X:list):
        assert len(X) == self.size, "dimensions of input data don't match the input of net"
        
        res = Var(0.0, "res")
        for i in range(self.size):
            res += self.weights[i] * X[i]
        res += self.bias
        res = res.tanh()
        return res

class Layer:
    def __init__(self, nins:int, nouts:int ):
        self.input_size = nins
        self.output_size = nouts
        self.perceptrons = [Perceptron(self.input_size) for _ in range(nouts)]
    
    def __call__(self, X:list):
        
        assert type(X) == list ,  "input data should be a list"
        res = [self.perceptrons[i](X) for i in range(self.output_size)]
        return res
    
class MLP:
    def __init__(self, dims):
        self.layers = [Layer(dims[i], dims[i+1]) for i in range(len(dims)-1)]
        self.dims = dims
        
    def __call__(self, X):
        assert type(X) == list ,  "input data should be a list"
        input_data = X
        output_data = None
        for i in range(len(self.layers)):
            output_data = self.layers[i](input_data)
            input_data = output_data
        return output_data 
    
    def weights(self):
        W = []
        for i in range(len(self.layers)):
            for j in range(len(self.layers[i].perceptrons)):
                    W += self.layers[i].perceptrons[j].weights
                    W += [self.layers[i].perceptrons[j].bias]
        return W
    
    def grads(self):
        G = []
        for i in range(len(self.layers)):
            for j in range(len(self.layers[i].perceptrons)):
                W = self.layers[i].perceptrons[j].weights
                G += [round(W[k].grad,2) for k in range(len(W))]
                G += [round(self.layers[i].perceptrons[j].bias.grad, 2)]
        return G