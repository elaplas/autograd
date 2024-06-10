# autograd

It is a small Python implementation of back-propagation in a fully connected neural network. The implementation of `autograd` is based on the following observations and illustrations:

- #### Calculation of gradient automatically by using back propagation
The gradient of loss function with respect to the weights of last layers are calculated and back propagated through the intermediate and first layers using chain rule. Given the following simple relations (1):

$y=w_1x_1+b_1$
$z=w_2y+b_2$

The derivatives $dz/dw_2$ , $dz/db_2$, $dz/dw_1$, and $dz/db_1$ are calculated as following:

$dz/dw_2 = y$
$dz/db_2 = 1$
$dz/dy = w_2$
$dz/dw_1 = dz/dy *  dy/dw_1 = w_2x_1$ 
$dz/db_1 = dz/dy *  dy/db_1 = w_2$
$dz/dx_1 = dz/dy *  dy/dx_1 = w_2 w_1$ 

The flowchart below illustrates the relations (1):
```mermaid
graph LR
A[w1] -->D(*) 
B[x1] -->D(*)
D(*) -->E(+)
C[b1] --> E(+)
E(+) --> |y| G(*)
M[w2] -->G(*)
N[b2] -->L(+)
G(*) -->L(+)
L(+) -->R[z]
```
There is a very helpful observation that the addition forwards the same derivative, and that the multiplication forwards the multiplication of the derivative with the other branch. This simplifies the implementation of back-propagation.

```mermaid
graph LR
A[dz/dw1 = w1*x1] -->D(*) 
B[dz/dx1 = w1*w2] -->D(*)
D(*) --> E(+)
C[dz/db1 = w2] --> E(+)
E(+) --> |dz/dy = w2| G(*)
M[dz/dw2 = y] -->G(*)
N[dz/db2=1] -->L(+)
G(*) --> L(+)
L(+) -->R[dz/dz=1]
```


- #### Multi layer perceptron (MLP)

A perceptron is a mathematical expression that squashes the weighted sum of inputs to the range [-1, 1], which has $n$ inputs but only one output.

```mermaid
graph LR
A[w1*x1] -->D(+) 
B[w2*x2] -->D(+)
M[wn*xn] -->D(+)
D(+) --> E(tanh)
E(tanh) -->G[y]
```
A hidden layer consists of $m$ perceptron each of which has a $n$ inputs. As a result a neural hidden layer has $n*m$ dimensions:


```mermaid
graph LR;
    subgraph hidden layer
        P1[P1]
        P2[P2]
        Pm[Pm]
    end

    P1 --> O
    P2 --> O
    Pm --> O
```

A multi layer perceptron consists of several layers, which are fully connected:

```mermaid
graph LR;
    subgraph Input Layer
        X1[X1]
        X2[X2]
        Xn[Xn]
    end

    subgraph Hidden Layer 1
        H1[H1]
        H2[H2]
        Hn[Hn]
    end

    subgraph Hidden Layer 2
        G1[H1]
        G2[H2]
    end

    subgraph Output Layer
        O1[O1]
        O2[O2]
    end

    X1 --> H1
    X1 --> H2
    X1 --> Hn

    X2 --> H1
    X2 --> H2
    X2 --> Hn

    Xn --> H1
    Xn --> H2
    Xn --> Hn

    H1 --> G1
    H1 --> G2

    H2 --> G1
    H2 --> G2

    Hn --> G1
    Hn --> G2

    G1 --> O1
    G2 --> O2
```
- #### Learning weights by gradient decent 

