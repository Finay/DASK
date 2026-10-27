import java.util.*;

public class Autograd {

    Tensor value;
    Tensor grad;
    Set<Autograd> _prev;
    String _op;
    Runnable _backward;
    boolean requiresGrad;

    public Autograd(Tensor value) {
        this(value, "", new Autograd[0]);
    }

    public Autograd(Tensor value, String _op, Autograd... children) {
        this.value = value;
        this._op = _op;
        this._prev = new HashSet<>(Arrays.asList(children));
        this._backward = () -> {};
        this.requiresGrad = value.requiresGrad;

        if (this.requiresGrad) {
            this.grad = Tensor.zeros(value.shape, false, false);
        }
    }


    public void setBackward(Runnable backward) {
        this._backward = backward;
    }


    public void addGrad(Tensor gradient) {
        if (!requiresGrad) {
            return;
        }

        else if (!Arrays.equals(this.value.shape, gradient.shape)) {
            throw new IllegalArgumentException("Gradient shape mismatch");
        }

        else {
            for (int i = 0; i < this.grad.data.length; i++) {
                this.grad.data[i] += gradient.data[i];
            }
        }
    }


    public void resetGrad() {
        if (this.grad != null) {
            for (int i = 0; i < this.grad.data.length; i++) {
                this.grad.data[i] = 0.0f;
            }
        }
    }


    private static void buildTopo(Autograd v, Set<Autograd> visited, List<Autograd> topo) {
        if (!visited.contains(v)) {
            visited.add(v);

            for (Autograd child : v._prev) {
                buildTopo(child, visited, topo);
            }
            topo.add(v);
        }
    }


    public void backward() {
        if (value.data.length != 1) {
            throw new IllegalStateException("Require an initial gradient");
        }

        Tensor gradient = Tensor.zeros(value.shape, false, false);
        gradient.data[0] = 1.0f;
        backward(gradient);
    }


    public void backward(Tensor gradient) {
        if (!requiresGrad) {
            throw new IllegalStateException("Tensor does not require gradient");
        }

        List<Autograd> topo = new ArrayList<>();
        Set<Autograd> visited = new HashSet<>();
        buildTopo(this, visited, topo);

        for (Autograd v : topo) {
            if (!v._prev.isEmpty()) {
                v.resetGrad();
            }
        }
        this.addGrad(gradient);

        for (int i = topo.size() - 1; i >= 0; i--) {
            Autograd v = topo.get(i);
            v._backward.run();
        }
    }


    public void zeroGrad() {
        List<Autograd> topo = new ArrayList<>();
        Set<Autograd> visited = new HashSet<>();
        buildTopo(this, visited, topo);

        for (Autograd v : topo) {
            v.resetGrad();
        }
    }
}
