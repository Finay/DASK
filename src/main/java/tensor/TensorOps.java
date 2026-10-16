package tensor;
import tensor.backend.BackendFactory;

//TODO: Implement the rest of these ops
public class TensorOps {

    private static void sameDevice(Tensor a, Tensor b){
        if(a.getDevice() != b.getDevice()){
            throw new InvalidDeviceException("Devices do not match.");
        }
    }

    public static Tensor add(Tensor a, Tensor b) {
        sameDevice(a, b);
        return BackendFactory.backendFor(a.getDevice()).add(a, b);
    }

    public static Tensor matmul(Tensor a, Tensor b) {
        sameDevice(a, b);
        return BackendFactory.backendFor(a.getDevice()).matmul(a, b);
    }
}