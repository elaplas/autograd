from var import Var

class Perceptron:
    def __init__(self, ninputs:int):
        self.size = ninputs
        self.weights = [Var(1.0, f"w {i}" ) for i in range(self.size)]
        self.bias = Var(1.0, "b")
    
    def __call__(self, X:list):
        assert len(X) == self.size, "dimensions of input data don't match the input of net"
        
        res = Var(0.0, "res")
        for i in range(self.size):
            res += self.weights[i] * X[i]
        res += self.bias
        res = res.tanh()
        return res
    
    
P1 = Perceptron(3)
res = P1([0.1,0.2,0.3])
res.backward()
print(res.data)
print(P1.weights[0].grad)
print(P1.weights[1].grad)
print(P1.weights[2].grad)
print(P1.bias.grad)