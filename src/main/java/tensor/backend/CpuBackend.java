package tensor.backend;

import tensor.Tensor;

public class CpuBackend implements TensorBackend {
        public Tensor matmul(Tensor a, Tensor b) {
        throw new UnsupportedOperationException("Not implemented yet");
    }

    public Tensor add(Tensor a, Tensor b) {
        throw new UnsupportedOperationException("Not implemented yet");
    }
    public Tensor subtract(Tensor a, Tensor b) {
        throw new UnsupportedOperationException("Not implemented yet");
    }
    public Tensor divide(Tensor a, Tensor b) {
        throw new UnsupportedOperationException("Not implemented yet");
    }
    public Tensor dot (Tensor a, Tensor b) {
        throw new UnsupportedOperationException("Not implemented yet");
    }
    public void add_(Tensor a, Tensor b) {
        throw new UnsupportedOperationException("Not implemented yet");
    }
    public void subtract_ (Tensor a, Tensor b) {
        throw new UnsupportedOperationException("Not implemented yet");
    }
}