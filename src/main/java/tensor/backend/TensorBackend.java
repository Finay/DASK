package tensor.backend;

import tensor.Tensor;

public interface TensorBackend {
    Tensor matmul(Tensor a, Tensor b);
    Tensor add(Tensor a, Tensor b);
    Tensor subtract(Tensor a, Tensor b);
    Tensor divide(Tensor a, Tensor b);
    double dot (Tensor a, Tensor b);
    void add_(Tensor a, Tensor b);
    void subtract_ (Tensor a, Tensor b);
}