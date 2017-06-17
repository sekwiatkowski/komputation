package shape.komputation.functions

import org.junit.jupiter.api.Assertions.assertArrayEquals
import org.junit.jupiter.api.Test
import shape.komputation.functions.activation.relu

class ReluTest {


    @Test
    fun test() {

        assertArrayEquals(
            relu(doubleArrayOf(-1.0, 0.0, 1.0)),
            doubleArrayOf(0.0, 0.0, 1.0))

    }


}