package tensor;

import tensor.backend.TensorBackend;
import tensor.backend.CpuBackend;
import tensor.backend.CudaBackend;


public class TensorOps {

    private static final TensorBackend cpu = new CpuBackend();
    private static final TensorBackend cuda = new CudaBackend();

    private static TensorBackend backendFor(Device device) {
        switch(device) {
            case CPU: return cpu;
            case CUDA: return cuda;
            default: throw new InvalidDeviceException(device + " is not a supported accelerator.");
        }
    }

    private static void sameDevice(Tensor a, Tensor b){
        if(a.getDevice() != b.getDevice()){
            throw new InvalidDeviceException("Devices do not match.");
        }
    }

    public static Tensor add(Tensor a, Tensor b) {
        sameDevice(a, b);
        return backendFor(a.getDevice()).add(a, b);
    }

    public static Tensor matmul(Tensor a, Tensor b) {
        sameDevice(a, b);
        return backendFor(a.getDevice()).matmul(a, b);
    }
}