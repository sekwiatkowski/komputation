package shape.komputation.cpu.optimization

import org.junit.jupiter.api.Assertions.assertArrayEquals
import org.junit.jupiter.api.Test

class DenseAccumulatorTest {

    @Test
    fun test() {

        val accumulator = DenseAccumulator(1)

        accumulator.accumulate(doubleArrayOf(1.0))

        val firstExpected = doubleArrayOf(1.0)
        val firstActual = accumulator.getAccumulation()

        assertArrayEquals(firstExpected, firstActual)

        accumulator.reset()

        accumulator.accumulate(doubleArrayOf(1.0))
        accumulator.accumulate(doubleArrayOf(2.0))

        val secondExpected = doubleArrayOf(3.0)
        val secondActual = accumulator.getAccumulation()

        assertArrayEquals(secondExpected, secondActual)

    }

}

