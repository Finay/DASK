package tensor;
import tensor.backend.*;
import java.util.Arrays;

public class TensorOps {

  private static void sameDevice(Tensor a, Tensor b) {
    if (a.getDevice() != b.getDevice()) {
      throw new InvalidDeviceException("Devices do not match.");
    }
  }

  private static void sameShape(Tensor a, Tensor b) {
    if (!Arrays.equals(a.getShape(), b.getShape())) {
      throw new IllegalArgumentException("Tensor shapes do not match.");
    }
  }

  public static Tensor add(Tensor a, Tensor b) {
    sameDevice(a, b);
    sameShape(a, b);

    Tensor out = BackendFactory.backendFor(a.getDevice()).add(a, b);

    if (a.requiresGrad() || b.requiresGrad()) {
      out.setRequiresGrad(true);

      out.setBackward(() -> {
        if (a.requiresGrad()) {
          a.addGrad(out.getGrad());
        }

        if (b.requiresGrad()) {
          b.addGrad(out.getGrad());
        }
      }, a, b);
    }

    return out;
  }

  public static Tensor subtract(Tensor a, Tensor b) {
    sameDevice(a, b);
    sameShape(a, b);

    Tensor out = BackendFactory.backendFor(a.getDevice()).subtract(a, b);

    if (a.requiresGrad() || b.requiresGrad()) {
      out.setRequiresGrad(true);

      out.setBackward(() -> {
        if (a.requiresGrad()) {
          a.addGrad(out.getGrad());
        }

            if (b.requiresGrad()) {
                TensorBackend backend = BackendFactory.backendFor(b.getDevice());
                Tensor zeros = Tensor.zeroes(b.getShape(), b.getDevice(), false);
                Tensor gradB = backend.subtract(zeros, out.getGrad());
                b.addGrad(gradB);
            }
        }, a, b);
    }

    return out;
  }

  public static Tensor divide(Tensor a, Tensor b) {
    sameDevice(a, b);
    sameShape(a, b);

    Tensor out = BackendFactory.backendFor(a.getDevice()).divide(a, b);

    if (a.requiresGrad() || b.requiresGrad()) {
      out.setRequiresGrad(true);

            out.setBackward(() -> {
                TensorBackend backend = BackendFactory.backendFor(a.getDevice());
                Tensor gradOut = out.getGrad();

                if (a.requiresGrad()) {
                    Tensor gradA = backend.divide(gradOut, b);
                    a.addGrad(gradA);
                }

                if (b.requiresGrad()) {
                    Tensor numerator = backend.multiply(gradOut, a);
                    Tensor denominator = backend.multiply(b, b);
                    Tensor quotient = backend.divide(numerator, denominator);
                    Tensor zeros = Tensor.zeroes(b.getShape(), b.getDevice(), false);
                    Tensor gradB = backend.subtract(zeros, quotient);
                    b.addGrad(gradB);
                }
            }, a, b);
        }

    return out;
  }

  public static Tensor matmul(Tensor a, Tensor b) {
    sameDevice(a, b);

    Tensor out = BackendFactory.backendFor(a.getDevice()).matmul(a, b);

    if (a.requiresGrad() || b.requiresGrad()) {
      out.setRequiresGrad(true);

      out.setBackward(() -> {
        if (a.requiresGrad()) {
          Tensor gradA = BackendFactory.backendFor(a.getDevice())
                  .matmul(out.getGrad(), b.transpose());

          a.addGrad(gradA);
        }

        if (b.requiresGrad()) {
          Tensor gradB = BackendFactory.backendFor(b.getDevice())
                  .matmul(a.transpose(), out.getGrad());

          b.addGrad(gradB);
        }
      }, a, b);
    }

    return out;
  }

  public static double dot(Tensor a, Tensor b) {
    sameDevice(a, b);
    sameShape(a, b);

    return BackendFactory.backendFor(a.getDevice()).dot(a, b);
  }

  public static void add_(Tensor a, Tensor b) {
    sameDevice(a, b);

    if (a.requiresGrad() || b.requiresGrad()) {
      throw new IllegalStateException(
              "In-place operations are not supported during autograd."
      );
    }

    BackendFactory.backendFor(a.getDevice()).add_(a, b);
  }

  public static void subtract_(Tensor a, Tensor b) {
    sameDevice(a, b);

    if (a.requiresGrad() || b.requiresGrad()) {
      throw new IllegalStateException(
              "In-place operations are not supported during autograd."
      );
    }

    BackendFactory.backendFor(a.getDevice()).subtract_(a, b);
  }


  public static Tensor multiply(Tensor a, Tensor b) {
    sameDevice(a, b);
    sameShape(a, b);

        TensorBackend backend = BackendFactory.backendFor(a.getDevice());
        Tensor out = backend.multiply(a, b);

        if (a.requiresGrad() || b.requiresGrad()) {
            out.setRequiresGrad(true);

            out.setBackward(() -> {
                if (a.requiresGrad()) {
                    Tensor gradA = backend.multiply(out.getGrad(), b);
                    a.addGrad(gradA);
                }

                if (b.requiresGrad()) {
                    Tensor gradB = backend.multiply(out.getGrad(), a);
                    b.addGrad(gradB);
                }
            }, a, b);
        }

    return out;
  }

    public static Tensor relu(Tensor a) {
        TensorBackend backend = BackendFactory.backendFor(a.getDevice());
        Tensor out = backend.relu(a);

        if (a.requiresGrad()) {
            out.setRequiresGrad(true);

            out.setBackward(() -> {
                Tensor gradA = backend.reluBackward(a, out.getGrad());
                a.addGrad(gradA);
            }, a);
        }

    return out;
  }

    public static Tensor mean(Tensor a) {
        TensorBackend backend = BackendFactory.backendFor(a.getDevice());
        Tensor out = backend.mean(a);

        if (a.requiresGrad()) {
            out.setRequiresGrad(true);

            out.setBackward(() -> {
                Tensor gradA = backend.meanBackward(a, out.getGrad());
                a.addGrad(gradA);
            }, a);
        }

    return out;
  }
}
