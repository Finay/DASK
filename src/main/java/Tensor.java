import java.util.stream.IntStream;
import java.util.Arrays;

enum Device {
    CUDA, CPU;
}

public class Tensor {

    float[] data;
    int[] shape;
    public int[] strides;
    boolean requiresGrad;
    boolean useCUDA;
    Device device = Device.CPU;

    private static final long PARALLEL_THRESH = 64L * 64 * 64;

    Tensor(int[] shape, float[] data, boolean useCUDA, boolean requiresGrad) {
        int flatDim = flatDimension(shape);
        if (data.length != flatDim) {
            throw new IllegalArgumentException(
                "Shape and Data mismatch : (shape flat dim) " + flatDim + 
                "!= " + data.length + " data length"
            );
        }
        this.data = data;
        this.shape = shape;
        this.useCUDA = useCUDA;
        this.requiresGrad = requiresGrad;
        this.strides = new int[shape.length];

        int acc = 1;
        for (int i = shape.length - 1; i >= 0; i--) {
            this.strides[i] = acc;
            acc *= shape[i];
        }

        if (useCUDA) { this.device = Device.CUDA;}
    }

    private Tensor(int[] shape, int[] strides, float[] data, boolean useCUDA, boolean requiresGrad) {
        this.shape = shape;
        this.strides = strides;
        this.data = data;
        this.useCUDA = useCUDA;
        this.requiresGrad = requiresGrad;
        if (useCUDA) { this.device = Device.CUDA;}

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

        return new Tensor(shapeCopy, stridesCopy, this.data, this.useCUDA,  this.requiresGrad);
    }

    public static Tensor zeros(int[] shape, boolean useCUDA, boolean requiresGrad) {
        float[] vals = new float[Tensor.flatDimension(shape)];
        return new Tensor(shape, vals, useCUDA, requiresGrad);
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

}