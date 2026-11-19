package tensor;

import org.junit.jupiter.api.Test;

import static org.junit.jupiter.api.Assertions.assertArrayEquals;
import static org.junit.jupiter.api.Assertions.assertThrows;

public class TensorTest {
  @Test
  void rejectsNonPositiveShape() {
    assertThrows(IllegalArgumentException.class,
            () -> Tensor.zeroes(new int[]{2, 0}, Device.CPU, false));
  }
}
