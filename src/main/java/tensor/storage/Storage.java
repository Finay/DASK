package tensor.storage;
import tensor.Device;

public interface Storage extends AutoCloseable {
    int size();
    Device device();
    void copyFromHost(float[] src);
    float[] toHostArray();

    float get(int i);
    void set(int i, float val);

    @Override 
    void close();
}
