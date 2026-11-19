1. Create a sin method that evaluates sin(x) 0<x<2pi
2. Network is 2 layers, m nodes each with a bias, input is a scalar x, output is a scalar y.
3. f(x)=sum(relu(C*relu(A*x+B)+D)) and loss is |sin(x)-f(x)|^2 each of A,B,C,D are tensors
4. Update params by inplace subtract each tensor by its gradient times a learning rate (which is a small scalar tensor)
5. Repeat until loss is low.
6. Demonstrate network behaves like sin