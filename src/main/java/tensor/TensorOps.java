package tensor;

import tensor.backend.BackendFactory;

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
          int[] shape = b.getShape();
          Tensor gradB = Tensor.zeroes(shape, b.getDevice(), false);

          int[] idx = new int[shape.length];
          int dim = Tensor.flatDimension(shape);

          for (int i = 0; i < dim; i++) {
            int remainder = i;

            for (int j = shape.length - 1; j >= 0; j--) {
              idx[j] = remainder % shape[j];
              remainder /= shape[j];
            }

            gradB.set(idx, -out.getGrad().get(idx));
          }

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
        int[] shape = out.getShape();
        int[] idx = new int[shape.length];
        int dim = Tensor.flatDimension(shape);

        Tensor gradA = a.requiresGrad()
                ? Tensor.zeroes(shape, a.getDevice(), false) : null;

        Tensor gradB = b.requiresGrad()
                ? Tensor.zeroes(shape, b.getDevice(), false) : null;

        for (int i = 0; i < dim; i++) {
          int remainder = i;

          for (int j = shape.length - 1; j >= 0; j--) {
            idx[j] = remainder % shape[j];
            remainder /= shape[j];
          }

          float x = a.get(idx);
          float y = b.get(idx);
          float g = out.getGrad().get(idx);

          if (a.requiresGrad()) {
            gradA.set(idx, g / y);
          }

          if (b.requiresGrad()) {
            gradB.set(idx, -g * x / (y * y));
          }
        }

        if (a.requiresGrad()) {
          a.addGrad(gradA);
        }

        if (b.requiresGrad()) {
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

    int[] shape = a.getShape();
    int dim = Tensor.flatDimension(shape);
    int[] idx = new int[shape.length];

    boolean requiresGrad = a.requiresGrad() || b.requiresGrad();
    Tensor out = Tensor.zeroes(shape, a.getDevice(), requiresGrad);

    for (int i = 0; i < dim; i++) {
      int remainder = i;

      for (int j = shape.length - 1; j >= 0; j--) {
        idx[j] = remainder % shape[j];
        remainder /= shape[j];
      }

      out.set(idx, a.get(idx) * b.get(idx));
    }

    if (requiresGrad) {
      out.setBackward(() -> {
        Tensor gradA = a.requiresGrad() ? Tensor.zeroes(shape, a.getDevice(), false) : null;
        Tensor gradB = b.requiresGrad() ? Tensor.zeroes(shape, b.getDevice(), false) : null;

        int[] idxGrad = new int[shape.length];

        for (int i = 0; i < dim; i++) {
          int remainder = i;

          for (int j = shape.length - 1; j >= 0; j--) {
            idxGrad[j] = remainder % shape[j];
            remainder /= shape[j];
          }

          float g = out.getGrad().get(idxGrad);

          if (a.requiresGrad()) {
            gradA.set(idxGrad, g * b.get(idxGrad));
          }

          if (b.requiresGrad()) {
            gradB.set(idxGrad, g * a.get(idxGrad));
          }
        }

        if (a.requiresGrad()) {
          a.addGrad(gradA);
        }

        if (b.requiresGrad()) {
          b.addGrad(gradB);
        }
      }, a, b);
    }

    return out;
  }

  public static Tensor relu(Tensor a) {
    int[] shape = a.getShape();
    int dim = Tensor.flatDimension(shape);
    int[] idx = new int[shape.length];

    Tensor out = Tensor.zeroes(shape, a.getDevice(), a.requiresGrad());

    for (int i = 0; i < dim; i++) {
      int remainder = i;

      for (int j = shape.length - 1; j >= 0; j--) {
        idx[j] = remainder % shape[j];
        remainder /= shape[j];
      }

      out.set(idx, Math.max(0.0f, a.get(idx)));
    }

    if (a.requiresGrad()) {
      out.setBackward(() -> {
        Tensor gradA = Tensor.zeroes(shape, a.getDevice(), false);
        int[] idxGrad = new int[shape.length];

        for (int i = 0; i < dim; i++) {
          int remainder = i;

          for (int j = shape.length - 1; j >= 0; j--) {
            idxGrad[j] = remainder % shape[j];
            remainder /= shape[j];
          }

          float g = a.get(idxGrad) > 0.0f
                  ? out.getGrad().get(idxGrad) : 0.0f;

          gradA.set(idxGrad, g);
        }

        a.addGrad(gradA);
      }, a);
    }

    return out;
  }


  public static Tensor mean(Tensor a) {
    int[] shape = a.getShape();
    int dim = Tensor.flatDimension(shape);
    int[] idx = new int[shape.length];

    float sum = 0.0f;

    for (int i = 0; i < dim; i++) {
      int remainder = i;

      for (int j = shape.length - 1; j >= 0; j--) {
        idx[j] = remainder % shape[j];
        remainder /= shape[j];
      }

      sum += a.get(idx);
    }

    Tensor out = new Tensor(
            new int[]{1},
            new float[]{sum / dim},
            a.getDevice(),
            a.requiresGrad()
    );

    if (a.requiresGrad()) {
      out.setBackward(() -> {
        Tensor gradA = Tensor.zeroes(shape, a.getDevice(), false);
        int[] idxGrad = new int[shape.length];

        float g = out.getGrad().get(new int[]{0}) / dim;

        for (int i = 0; i < dim; i++) {
          int remainder = i;

          for (int j = shape.length - 1; j >= 0; j--) {
            idxGrad[j] = remainder % shape[j];
            remainder /= shape[j];
          }

          gradA.set(idxGrad, g);
        }

        a.addGrad(gradA);
      }, a);
    }

    return out;
  }
}
