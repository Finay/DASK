package tensor.backend;

import tensor.Tensor;

public interface TensorBackend {
    Tensor matmul(Tensor a, Tensor b);

    Tensor add(Tensor a, Tensor b);

    Tensor subtract(Tensor a, Tensor b);

    Tensor divide(Tensor a, Tensor b);

    Tensor multiply(Tensor a, Tensor b);

    Tensor relu(Tensor a);

    Tensor reluBackward(Tensor a, Tensor gradOut);

    Tensor mean(Tensor a);

    Tensor meanBackward(Tensor a, Tensor gradOut);

    double dot(Tensor a, Tensor b);

    void add_(Tensor a, Tensor b);

    void subtract_(Tensor a, Tensor b);
}