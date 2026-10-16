package tensor.backend;
import tensor.Device;

public class BackendFactory {

    private static final TensorBackend cudaBackend = new CudaBackend();
    private static final TensorBackend cpuBackend = new CpuBackend();



    public static TensorBackend backendFor(Device device) {
        switch (device){
            case CPU: return cpuBackend;
            case CUDA: return cudaBackend;
            default: throw new IllegalArgumentException("Invalid Device");
        }
    }
}
