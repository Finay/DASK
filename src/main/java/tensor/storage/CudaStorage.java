package tensor.storage;

import tensor.Device;

//TODO: Impmenet this file

public class CudaStorage implements Storage {

    private final float[] data;
    public CudaStorage(int size) { this.data = new float[size];}
    public int size() {return data.length;}
    public Device device() {return Device.CUDA;}
    public float[] toHostArray() {return data.clone();}
    public void close() {}
    public float get(int i) {return data[i];}
    public void set(int i, float val) {data[i] = val;}
    public float[] floats() {return data;}

    public void copyFromHost(float[] src) {
        if (src.length != data.length) {
            throw new IllegalArgumentException("Length Mistmatch");
        } 
        System.arraycopy(src, 0, data, 0, src.length);
    }

}
