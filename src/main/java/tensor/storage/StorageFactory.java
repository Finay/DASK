package tensor.storage;

import tensor.Device;

public class StorageFactory {
    public static Storage alloc(Device device, int size) {
        switch (device){
            case CPU: return new CpuStorage(size);
            case CUDA: return new CudaStorage(size);
            default: throw new IllegalArgumentException("Invalid Device");
        }
    }
}
