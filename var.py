import math


class Var:
    def __init__(self, val, name=""):
        self.data = val
        self.name = name
        self.grad = 0.0
        self._grad_func = None
        self._children = set()
    
    def __repr__(self):
        return "{:.2f}".format(self.data)
        
    def __add__(self, other):
        if type(other) != Var:
            other = Var(other)
        res = Var(self.data + other.data, "res")
        def grad():
            # The gradients are accumulated because there could be contributions from
            # many outputs if the same variable is used in different addition operations
            # e.g. build computation graph with multiple paths for following operations
            # y = x + a       # Path 1: x is used here
            # z = x + b       # Path 2: x is used again here
            # result = y + z  # Final output
            self.grad += res.grad
            other.grad += res.grad
        res._grad_func = grad
        res._children.add(self)
        res._children.add(other)
        return res
    
    ## This function handles the cases like w*x - 4, which is w*x + (-1) * 4 where 4 is b. The derivate w.r.t b is -1.  
    ## This is different than the cases like w*x + (-4) where -4 is b. The derivate w.r.t b is 1. (This can is handled by __add__())
    def __sub__(self, other):
        if type(other) != Var:
            other = Var(other)
        res = Var(self.data - other.data, "res")
        def grad():
            # The gradients are accumulated because there could be contributions from
            # many outputs if the same variable is used in different addition operations
            self.grad += res.grad
            other.grad += -1*res.grad
        res._grad_func = grad
        res._children.add(self)
        res._children.add(other)
        return res
            
        
    def __mul__(self, other):
        if type(other) != Var:
            other = Var(other)
        res = Var(self.data * other.data, "res")
        def grad():
            # The gradients are accumulated because there could be contributions from
            # many outputs if the same variable is used in different addition operations
            self.grad += other.data * res.grad
            other.grad += self.data * res.grad
        res._grad_func = grad
        res._children.add(self)
        res._children.add(other)
        return res
    
    def __pow__(self, n):
        res = Var(self.data**n, "res")
        def grad():
            # Gradient of "self" is assumed to be "1" because this "pow" is topmost function
            # in the loss if the same variable is used in different addition operations
            self.grad += n*((self.data)**(n-1))*1
        res._grad_func = grad
        res._children.add(self)
        return res   
    
    def tanh(self):
        x = self.data
        t = (math.exp(2*x)-1)/(math.exp(2*x)+1)
        res = Var(t, "res")
        def grad():
            # The gradient is accumulated because there could be contributions from
            # many outputs if the same variable is used in different addition operations
            self.grad += (1 - (res.data**2))*res.grad
        res._grad_func = grad
        res._children.add(self)
        return res
    
    def __rmul__(self, other):
        return self*other
    
    def __radd__(self, other):
        return self+other
    
    def backward(self):
        if self._grad_func is None:
            print("the variable is not derivable")
            return None

        self.grad = 1
        nodes = [self]
        visited = set()
        visited.add(self)
        while len(nodes):
            cur_node = nodes[0]
            if cur_node._grad_func is not None:
                cur_node._grad_func()
                for child in cur_node._children:
                    if child not in visited:
                        nodes.append(child)
                        visited.add(child)
            nodes.pop(0)