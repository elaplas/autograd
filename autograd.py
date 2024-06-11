class Var:
    def __init__(self, val, name=""):
        self.data = val
        self.name = name
        self.grad = 1.0
        self._grad_func = None
        self._children = set()
        
    def __add__(self, other):
        if type(other) != Var:
            other = Var(other)
        res = Var(self.data + other.data, "res")
        def grad():
            self.grad = res.grad
            other.grad = res.grad
        res._grad_func = grad
        res._children.add(self)
        res._children.add(other)
        return res
        
    def __mul__(self, other):
        if type(other) != Var:
            other = Var(other)
        res = Var(self.data * other.data, "res")
        def grad():
            self.grad = other.data * res.grad
            other.grad = self.data * res.grad
        res._grad_func = grad
        res._children.add(self)
        res._children.add(other)
        return res   
    
    def backward(self):
        if self._grad_func is None:
            print("the variable is not derivable")
            return None
    
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
                             

w1 = Var(3, "w1")
x1 = Var(2, "x1")
b1 = Var(1, "b1")
w2 = Var(4, "w2")
b2 = Var(7, "b2")

b1.name = "b1"
y = (w1*x1) + b1
z = (y*w2) + b2

z.backward()

print(w1.name, ": ", w1.grad)
print(x1.name, ": ", x1.grad)
print(b1.name, ": ", b1.grad)
print(y.name, ": ", y.grad)
print(w2.name, ": ", w2.grad)
print(b2.name, ": ", b2.grad)