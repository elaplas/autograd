# autograd

It is a small Python implementation of back-propagation in a fully connected neural network. The implementation of `autograd` is based on the following observations and illustrations:

- ### Calculation of gradient automatically by using back propagation
The gradient of loss function with respect to the weights of last layers are calculated and back propagated through the intermediate and first layers using chain rule. Given the following simple relations (1):

$y=w_1x_1+b_1$
$g=sig(y)$
$z=w_2g+b_2$

The derivatives $dz/dw_2$ , $dz/db_2$, $dz/dw_1$, and $dz/db_1$ are calculated as following:

\[
\frac{dz}{dw_2} = \text{sig}(y)
\]
\[
\frac{dz}{db_2} = 1
\]
\[
\frac{dz}{dg} = w_2
\]
\[
\frac{dz}{dy} = \frac{dz}{dg} \cdot \frac{dg}{dy} = w_2 \cdot \text{sig}(y) \cdot (1 - \text{sig}(y))
\]
\[
\frac{dz}{dw_1} = \frac{dz}{dg} \cdot \frac{dg}{dy} \cdot \frac{dy}{dw_1} = w_2 \cdot \text{sig}(y) \cdot (1 - \text{sig}(y)) \cdot x_1
\]
\[
\frac{dz}{db_1} = \frac{dz}{dg} \cdot \frac{dg}{dy} \cdot \frac{dy}{db_1} = w_2 \cdot \text{sig}(y) \cdot (1 - \text{sig}(y))
\]


The flowchart below illustrates the relations (1):
```mermaid
graph LR
A[w1] -->D('*') 
B[x1] -->D('*')
D('*') -->E('+')
C[b1] --> E('+')
E('+') --> |y| S[sig]
S[sig] --> |g| G('*')
M[w2] -->G('*')
N[b2] -->L('+')
G('*') -->L('+')
L('+') -->R[z]
```
There is a very helpful observation that the addition node passes the previous derivative to its child nodes, and that the multiplication node passes the result of the multiplication of the previous derivative with the value of the other child node. This simplifies the implementation of back-propagation.

```mermaid
graph LR
A[w1] --> |x1.sig.1-sig.w_2| D('*') 
B[x1] --> |w1.sig.1-sig.w_2| D('*')
D('*') --> |sig . 1-sig . w_2| E('+')
E('+') --> |sig . 1-sig . w_2 | S[sig . 1-sig]
S[sig . 1-sig ] --> |w_2 . 1| J[g]
J[g] --> | w_2 . 1| G('*')
M[w_2] --> |g . 1| G('*')
N[b2]--> |1| L('+')
G('*') --> |1| L('+')
L('+') -->R[dz/dz=1]
```


- ### Multi layer perceptron (MLP)

A perceptron is a mathematical expression that squashes the weighted sum of inputs $X$ to the range [-1, 1], which has $n$ inputs but only one output.

```mermaid
graph LR
A[w1*x1] -->D('+') 
B[w2*x2] -->D('+')
M[wn*xn] -->D('+')
D('+') --> E(tanh)
E(tanh) -->G[y]
```
A hidden layer consists of $m$ perceptron each of which has a $n$ inputs. As a result a neural hidden layer has $n*m$ dimensions:


```mermaid
graph LR;

    subgraph Input Layer
        X1[X]
    end

    subgraph hidden layer
        P1[P1]
        P2[P2]
        Pm[Pm]
    end

    subgraph output layer
        O1[o1]
        O2[o2]
        Om[om]
    end

    X1 --> P1
    X1 --> P2
    X1 --> Pm

    P1 --> O1
    P2 --> O2
    Pm --> Om
```

A multi layer perceptron (MLP) consists of several layers, which are fully connected. The output of MLP is a function of data point $X_i$ with $n$ dimensions and weights $W$:

\( \hat{Y}_i = f(X_i, W) \)


```mermaid
graph LR;
    subgraph Input Layer
        X1[Xi]
    end

    subgraph Hidden Layer 1
        H1[P1]
        H2[P2]
        Hn[Pm]
    end

    subgraph Hidden Layer 2
        G1[P1]
        G2[P2]
    end

    subgraph Output Layer
        O1[o1]
        O2[o2]
    end

    X1 --> H1
    X1 --> H2
    X1 --> Hn

    H1 --> G1
    H1 --> G2

    H2 --> G1
    H2 --> G2

    Hn --> G1
    Hn --> G2

    G1 --> O1
    G2 --> O2
```
- ### Learning weights by gradient decent 

Given $N$ number of data points $X_i$, $N$ number of measurements $Y_i$, and a MLP, that is abstracted as \( f(X_i, W) \), the following loss function is defined:

\[ \text{L} = \frac{1}{N}\sum_{i=1}^{n} (Y_i - \hat{Y}_i)^2 \]

Where:
- \( n \) is the number of data points with $m$ dimensions
- \( Y_i \) is the observation for the \( i \)-th data point.
- \( \hat{Y}_i \) is the prediction for the \( i \)-th data point, which can be assumed that is the output of a multi layer perceptron, \( \hat{Y}_i = f(X_i, W) \). 

In the loss function $L$, the data points $X_i$ vary but the weights $W$ are constant. The weights should be tuned in a way that the predictions \( \hat{Y}_i\) are close enough to $Y_i$ so that $L$ goes to zero. The weights are iteratively tuned using gradient descent:

\[ W_{k+1} = W_{k} - \alpha \nabla_{W} L(W_{k}) \]

- \( W \) represents the parameters (weights) of the model.
- \( \alpha \) is the learning rate, a positive scalar that controls the step size.
- \( \nabla_{W} L(W) \) is the gradient of the function \( f \) with respect to \( W \) at \( W_k \) .

The gradient of loss function $L$ with respect to weights $W$ is calculated using the back-propagation explained above. The number of data points could be too many leading to a very big loss function and consequently too much sequential computation at the time of gradient computation. Therefore the data points are split in batches so that they are independently substituted in loss function and the corresponding gradient is calculated separately. This makes loss function smaller and enables parallel tunning of weights.