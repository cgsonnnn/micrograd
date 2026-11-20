import matplotlib as math

class Value:
    """ stores a single scalar value and its gradient """

    def __init__(self, data, _children=(), _op=''):
        self.data = data
        self.grad = 0
        # internal variables used for autograd graph construction
        self._backward = lambda: None
        self._prev = set(_children)
        self._op = _op # the op that produced this node, for graphviz / debugging / etc

    def __add__(self, other):
        other = other if isinstance(other,Value) else Value(other)
        out = Value(self.data + other.data, (self, other), '+')

        def _backward():
            self.grad += out.grad
            other.grad += out.grad
        out._backward = _backward

        return out

    def __radd__(self, other):
        return self + other

    def __rmul__(self, other):
        return self * other

    def __mul__(self, other):
        other = other if isinstance(other, Value) else Value(other)
        out = Value(self.data * other.data, (self, other), '*')

        def _backward():
            self.grad += other.data * out.grad
            other.grad += self.data * out.grad
        out._backward = _backward

        return out
    def __pow__(self,constant):
        out = Value(self.data**constant,(self,),"^")
        def _backward():
            self.grad += self.data**(constant - 1) * constant *out.grad
        out._backward = _backward
        return out

    def __neg__(self):
        out = Value(-self.data,(self,),"--")
        def _backward():
            self.grad += -out.grad
        out._backward = _backward
        return out

    def __sub__(self,other):
        return self + (-other)

    def __truediv__(self,other):
        return self * other**(-1)

    def exp(self):
        out = Value(math.exp(self.data),(self,),"e**")
        def _backward():
            self.grad += out.data * out.grad
        out._backward = _backward
        return out

    def backward(self):

        # topological order all of the children in the graph
        topo = []
        visited = set()
        def build_topo(v):
            if v not in visited:
                visited.add(v)
                for child in v._prev:
                    build_topo(child)
                topo.append(v)
        build_topo(self)

        # go one variable at a time and apply the chain rule to get its gradient
        self.grad = 1
        for v in reversed(topo):
            v._backward()
    
def numgrad(f,inputs,h=1e-6):
    lis = []
    base = f(*inputs)
    for i in range(len(inputs)):
        st = list(inputs)
        st[i] += h
        lis.append((f(*st)-base)/h)
    return lis

def f(a,b,c):
    return (a*b+c)*(a+c)

def grad_des(round,w,lr):
    w = Value(w)
    for i in range(round):
        w.grad = 0    
        loss = (w-3)**2
        loss.backward()
        print(loss.data,w.data,w.grad)
        w.data -= w.grad * lr


def grad_des2(round,a,b,lr):    
    a = Value(a)
    b = Value(b)
    for i in range(round):
        a.grad = 0
        b.grad = 0
        loss = (a * b - 6)**2 + (a + b - 5)**2
        loss.backward()
        a.data -= a.grad * lr
        b.data -= b.grad * lr
    print(loss.data,a.data,b.data,a.grad,b.grad)