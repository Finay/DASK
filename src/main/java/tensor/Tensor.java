
package tensor;

import tensor.storage.Storage;
import tensor.storage.StorageFactory;
import tensor.backend.BackendFactory;

import java.util.*;


public class Tensor {

    private Storage storage;
    private int[] shape;
    private int[] strides;
    private boolean requiresGrad;
    private Tensor grad;
    private Set<Tensor> parents;
    private Runnable backwardFunction;

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
        this.strides = continguousStrides(shape);
        this.requiresGrad = requiresGrad;
        this.grad = null;
        this.parents = new HashSet<>();
        this.backwardFunction = () -> {
        };

    }

    private Tensor(int[] shape, int[] strides, Storage storage, boolean requiresGrad) {
        this.shape = shape;
        this.strides = strides;
        this.storage = storage;
        this.requiresGrad = requiresGrad;
        this.grad = null;
        this.parents = new HashSet<>();
        this.backwardFunction = () -> {
        };

    }

    private static int[] continguousStrides(int[] shape) {
        int acc = 1;
        int[] strides = new int[shape.length];
        for (int i = shape.length - 1; i >= 0; i--) {
            strides[i] = acc;
            acc *= shape[i];
        }
        return strides;
    }


    public Tensor transpose() {
        int right = shape.length - 1;
        int left = 0;

        int[] shapeCopy = this.shape.clone();
        int[] stridesCopy = this.strides.clone();

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

        Tensor out = new Tensor(shapeCopy, stridesCopy, this.storage, this.requiresGrad);

        if (this.requiresGrad) {
            out.setBackward(() -> this.addGrad(out.getGrad().transpose()), this);
        }

        return out;
    }

    public static Tensor zeroes(int[] shape, Device device, boolean requiresGrad) {
        Storage storage = StorageFactory.alloc(device, flatDimension(shape));
        return new Tensor(shape.clone(), continguousStrides(shape), storage, requiresGrad);
    }

    public static int flatDimension(int[] shape) {
        int dim = 1;
        for (int i : shape) {
            if (i <= 0) {
                throw new IllegalArgumentException("Shape cannot contain non-positive values.");
            }
            dim *= i;
        }
        return dim;
    }

    private int flatIndex(int[] idx) {
        int off = 0;
        for (int i = 0; i < idx.length; i++) {
            off += idx[i] * this.strides[i];
        }
        return off;
    }

    public void set(int[] idx, float val) {
        this.storage.set(flatIndex(idx), val);
    }

    public float get(int[] idx) {
        return this.storage.get(flatIndex(idx));
    }

    public int[] getStrides() {
        return this.strides.clone();
    }

    public int[] getShape() {
        return this.shape.clone();
    }

    public Storage getStorage() {
        return this.storage;
    }

    public Device getDevice() {
        return this.storage.device();
    }

    public boolean requiresGrad() {
        return this.requiresGrad;
    }

    public Tensor getGrad() {
        return this.grad;
    }

    public void setRequiresGrad(boolean requiresGrad) {
        this.requiresGrad = requiresGrad;
    }

    public void setBackward(Runnable backwardFunction, Tensor... parents) {
        if (!this.requiresGrad) {
            return;
        }

        this.parents = new HashSet<>(Arrays.asList(parents));
        this.backwardFunction = backwardFunction;
    }

    public void addGrad(Tensor gradient) {
        if (!this.requiresGrad) {
            return;
        }

        if (!Arrays.equals(this.shape, gradient.getShape())) {
            throw new IllegalArgumentException("Gradient shape mismatch.");
        }

        if (this.grad == null) {
            this.grad = Tensor.zeroes(this.shape, this.getDevice(), false);
        }

        BackendFactory.backendFor(this.getDevice()).add_(this.grad, gradient);
    }

    private void resetGrad() {
        if (this.grad == null) {
            return;
        }

        this.grad = Tensor.zeroes(this.shape, this.getDevice(), false);
    }

    private static void buildTopo(
            Tensor tensor,
            Set<Tensor> visited,
            List<Tensor> topo) {

        if (!visited.contains(tensor)) {
            visited.add(tensor);

            for (Tensor parent : tensor.parents) {
                buildTopo(parent, visited, topo);
            }

            topo.add(tensor);
        }
    }

    public void backward() {
        if (flatDimension(this.shape) != 1) {
            throw new IllegalStateException("Require an initial gradient.");
        }

        Tensor gradient = Tensor.zeroes(this.shape, this.getDevice(), false);
        gradient.set(new int[this.shape.length], 1.0f);

        backward(gradient);
    }

    public void backward(Tensor gradient) {
        List<Tensor> topo = new ArrayList<>();
        Set<Tensor> visited = new HashSet<>();
        buildTopo(this, visited, topo);

        for (Tensor tensor : topo) {
            if (!tensor.parents.isEmpty()) {
                tensor.resetGrad();
            }
        }

        this.addGrad(gradient);

        for (int i = topo.size() - 1; i >= 0; i--) {
            Tensor tensor = topo.get(i);
            tensor.backwardFunction.run();
        }
    }

    public void zeroGrad() {
        List<Tensor> topo = new ArrayList<>();
        Set<Tensor> visited = new HashSet<>();
        buildTopo(this, visited, topo);

        for (Tensor tensor : topo) {
            tensor.resetGrad();
        }
    }
}