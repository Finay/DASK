
package tensor;
import tensor.storage.Storage;
import tensor.storage.StorageFactory;

public class Tensor {

    private Storage storage;
    private int[] shape;
    private int[] strides;
    private boolean requiresGrad;

    Tensor(int[] shape, float[] data, Device device, boolean requiresGrad) {
        int flatDim = flatDimension(shape);
        if (data.length != flatDim) {
            throw new IllegalArgumentException(
                "Shape and Data mismatch : (shape flat dim) " + flatDim + 
                "!= " + data.length + " data length"
            );
        }

        this.storage = StorageFactory.alloc(device, flatDim);
        this.storage.copyFromHost(data);
        this.shape = shape.clone();
        this.strides = new int[shape.length];
        this.requiresGrad = requiresGrad;

        int acc = 1;
        for (int i = shape.length - 1; i >= 0; i--) {
            this.strides[i] = acc;
            acc *= shape[i];
        }
    }

    private Tensor(int[] shape, int[] strides, Storage storage, boolean requiresGrad) {
        this.shape = shape;
        this.strides = strides;
        this.storage = storage;
        this.requiresGrad = requiresGrad;

    }

    public Tensor transpose() {
        int right = shape.length - 1;
        int left = 0;

        int[] shapeCopy = this.shape.clone();
        int [] stridesCopy = this.strides.clone();

        while (right > left) {
            int tempShape = shapeCopy[right];
            shapeCopy[right] = shapeCopy[left];
            shapeCopy[left] = tempShape;

            int tempStride = stridesCopy[right];
            stridesCopy[right] = stridesCopy[left];
            stridesCopy[left] = tempStride;

            left++;
            right--;
        }

        return new Tensor(shapeCopy, stridesCopy, this.storage, this.requiresGrad);
    }

    public static Tensor zeroes(int[] shape, Device device, boolean requiresGrad) {
        float[] vals = new float[Tensor.flatDimension(shape)];
        return new Tensor(shape, vals, device, requiresGrad);
    }

    private static int flatDimension(int[] shape) {
        int dim = 1;
        for (int i : shape){ 
            if (i <= 0) { throw new IllegalArgumentException("Shape cannot contain non-positive values.");}
            dim *= i;
        } return dim;
    }

    private int flatIndex(int[] idx) {
        int off = 0;
        for (int i = 0; i < idx.length; i++) {off += idx[i] * this.strides[i];}
        return off;
    }

    public void set(int[] idx, float val) {this.storage.set(flatIndex(idx), val);}
    public float get(int[] idx) {return this.storage.get(flatIndex(idx));}
    public int[] getStrides() { return this.strides.clone();}
    public int[] getShape() {return this.shape.clone();}
    public Storage getStorage() {return this.storage;}
    public Device getDevice() {return this.storage.device();}
    public boolean requiresGrad() {return this.requiresGrad;}

}