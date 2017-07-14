package shape.komputation.cpu.functions.activation

import org.junit.jupiter.api.Assertions.assertArrayEquals
import org.junit.jupiter.api.Test

class ReluTest {

    @Test
    fun test() {

        assertArrayEquals(
            relu(doubleArrayOf(-1.0, 0.0, 1.0)),
            doubleArrayOf(0.0, 0.0, 1.0))

    }

}