package shape.komputation.cpu.functions.activation

import org.junit.jupiter.api.Assertions.assertArrayEquals
import org.junit.jupiter.api.Test

class ReluTest {

    @Test
    fun test() {

        assertArrayEquals(
            relu(floatArrayOf(-1.0f, 0.0f, 1.0f)),
            floatArrayOf(0.0f, 0.0f, 1.0f))

    }

}