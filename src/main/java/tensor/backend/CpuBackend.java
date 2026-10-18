package tensor.backend;

import tensor.Tensor;
import tensor.Device;
import tensor.storage.CpuStorage;

import java.util.ArrayList;
import java.util.List;
import java.util.Arrays;
import java.util.concurrent.ExecutionException;
import java.util.concurrent.ExecutorService;
import java.util.concurrent.Callable;
import java.util.concurrent.Future;

import java.util.concurrent.Executors;

public class CpuBackend implements TensorBackend {
    
    private final int numThreads;
    private final ExecutorService pool;
    private static final int SERIAL_THRESH = 32_768;

    @FunctionalInterface 
    private interface RangeTask {void run(int start, int end);}

    /**
     * Initializes the CPU Backend
     * 
     * For debugging, reference java docs on concurrency, thread pools, lambdas.
     */
    public CpuBackend() {
        this.numThreads = Runtime.getRuntime().availableProcessors();
        this.pool = Executors.newFixedThreadPool(numThreads, r -> {
            Thread t = new Thread(r, "cpu-backend");
            t.setDaemon(true);
            return t;
        });
    }

    private void parallelFor(int n, RangeTask task) {
        parallelFor(n, n, task);
    }
 
    private void parallelFor(int n, long work, RangeTask task) {
        if (work < SERIAL_THRESH || n < 2) {
            task.run(0, n);
            return;
        }

        int chunks = Math.min(numThreads, n);
        List<Callable<Void>> jobs = new ArrayList<>(chunks);

        for(int c = 0; c < chunks; c++) {
            int start = (int)((long) n * c / chunks);
            int end = (int)((long) n * (c + 1) / chunks);
            jobs.add(() -> {task.run(start,end); return null;});
        }

        try {
            for (Future<Void> f : pool.invokeAll(jobs)) {
                f.get();
            }
        } catch (InterruptedException e) {
            Thread.currentThread().interrupt();
            throw new RuntimeException(e);
        } catch (ExecutionException e) {
            throw new RuntimeException(e);
        }
    }

    private static float[] floats (Tensor tens) {
        return ((CpuStorage) tens.getStorage()).floats();
    }

    private static void checkElementwise(Tensor a, Tensor b) {
        if (!Arrays.equals(a.getShape(), b.getShape())) {
            throw new IllegalArgumentException("Shape Mistmatch");
        }
    }
    
    public Tensor matmul(Tensor a, Tensor b) {
        int[] as = a.getShape();
        int[] bs = b.getShape();

        if(as.length != 2 || bs.length != 2 || as[1] != bs[0]) {
            throw new IllegalArgumentException("Shape Mismatch");
        }

        int M = as[0];
        int K = as[1];
        int N = bs[1];

        Tensor c = Tensor.zeroes(new int[]{M, N}, Device.CPU, a.requiresGrad() || b.requiresGrad());

        float[] x = floats(a);
        float[] y = floats(b);
        float[] z = floats(c);

        int[] sa = a.getStrides();
        int[] sb = b.getStrides();

        int sa0 = sa[0];
        int sa1 = sa[1];
        int sb0 = sb[0];
        int sb1 = sb[1];

        parallelFor(M, (long) M * N * K, (s, e) -> {
            for(int i = s; i < e; i++) {
                int rowC = i * N;
                for (int k = 0; k < K; k++) {
                    float aik = x[i * sa0 + k * sa1];
                    int rowB = k * sb0;
                    for(int j = 0; j < N; j++) {
                        z[rowC + j] += aik * y[rowB + j * sb1];
                    }
                }
            }
        });

        return c;
    }

    public Tensor add(Tensor a, Tensor b) {
        checkElementwise(a, b);
        Tensor c = Tensor.zeroes(a.getShape(), Device.CPU, a.requiresGrad() || b.requiresGrad());

        float[] x = floats(a);
        float[] y = floats(b);
        float[] z = floats(c);

        parallelFor(z.length, (s, e) -> {
            for (int i = s; i < e; i++) {
                z[i] = x[i] + y[i];
            }
        }); 

        return c;
    }
    public Tensor subtract(Tensor a, Tensor b) {
        checkElementwise(a, b);
        Tensor c = Tensor.zeroes(a.getShape(), Device.CPU, a.requiresGrad() || b.requiresGrad());

        float[] x = floats(a);
        float[] y = floats(b);
        float[] z = floats(c);

        parallelFor(z.length, (s, e) -> {
            for (int i = s; i < e; i++) {
                z[i] = x[i] - y[i];
            }
        }); 

        return c;
    }
    public Tensor divide(Tensor a, Tensor b) {
        checkElementwise(a, b);
        Tensor c = Tensor.zeroes(a.getShape(), Device.CPU, a.requiresGrad() || b.requiresGrad());

        float[] x = floats(a);
        float[] y = floats(b);
        float[] z = floats(c);

        parallelFor(z.length, (s, e) -> {
            for (int i = s; i < e; i++) {
                z[i] = x[i] / y[i];
            }
        }); 

        return c;
    }
    public double dot (Tensor a, Tensor b) {
        checkElementwise(a, b);
        Tensor c = Tensor.zeroes(a.getShape(), Device.CPU, a.requiresGrad() || b.requiresGrad());

        float[] x = floats(a);
        float[] y = floats(b);
        float[] z = floats(c);
        final double[] acc = {0.0};

        parallelFor(z.length, (s, e) -> {
            double local = 0.0;
            for (int i = s; i < e; i++) {
                local += x[i] * y[i];
            }
            synchronized (acc) {
                acc[0] += local;
            }
        }); 

        return acc[0];
    }
    public void add_(Tensor a, Tensor b) {
        checkElementwise(a, b);

        float[] x = floats(a);
        float[] y = floats(b);

        parallelFor(x.length, (s, e) -> {
            for (int i = s; i < e; i++) {
                x[i] += y[i];
            }
        }); 
    }
    public void subtract_ (Tensor a, Tensor b) {
        checkElementwise(a, b);

        float[] x = floats(a);
        float[] y = floats(b);

        parallelFor(x.length, (s, e) -> {
            for (int i = s; i < e; i++) {
                x[i] -= y[i];
            }
        }); 
    }
}