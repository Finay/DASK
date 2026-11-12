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

    public static Tensor subtract(Tensor a, Tensor b) {
        sameDevice(a, b);
        return BackendFactory.backendFor(a.getDevice()).subtract(a, b);
    }

    public static Tensor divide(Tensor a, Tensor b) {
        sameDevice(a, b);
        return BackendFactory.backendFor(a.getDevice()).divide(a, b);
    }

    public static double dot(Tensor a, Tensor b) {
        sameDevice(a, b);
        return BackendFactory.backendFor(a.getDevice()).dot(a, b);
    }

    public static void add_(Tensor a, Tensor b) {
        sameDevice(a, b);
        BackendFactory.backendFor(a.getDevice()).add_(a, b);
    }

    public static void subtract_(Tensor a, Tensor b) {
        sameDevice(a, b);
        BackendFactory.backendFor(a.getDevice()).subtract_(a, b);
    }
}