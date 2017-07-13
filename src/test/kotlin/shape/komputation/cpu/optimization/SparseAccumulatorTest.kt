package shape.komputation.cpu.optimization

import org.junit.jupiter.api.Assertions.assertArrayEquals
import org.junit.jupiter.api.Assertions.assertEquals
import org.junit.jupiter.api.Test

class SparseAccumulatorTest {

    @Test
    fun testOneDimension() {

        val accumulator = SparseAccumulator(1, 1, 1, 1)

        accumulator.accumulate(intArrayOf(0), doubleArrayOf(1.0))

        check(accumulator, 1, intArrayOf(0), doubleArrayOf(1.0), arrayOf(doubleArrayOf(1.0)))

        accumulator.accumulate(intArrayOf(0), doubleArrayOf(2.0))

        check(accumulator, 1, intArrayOf(0), doubleArrayOf(2.0), arrayOf(doubleArrayOf(3.0)))

    }

    @Test
    fun testTwoDimensions() {

        val accumulator = SparseAccumulator(1, 1, 1, 2)

        accumulator.accumulate(intArrayOf(0), doubleArrayOf(1.1, 1.2))

        check(accumulator, 1, intArrayOf(0), doubleArrayOf(1.0), arrayOf(doubleArrayOf(1.1, 1.2)))

        accumulator.accumulate(intArrayOf(0), doubleArrayOf(1.0, 2.0))

        check(accumulator, 1, intArrayOf(0), doubleArrayOf(2.0), arrayOf(doubleArrayOf(2.1, 3.2)))

    }

    @Test
    fun testTwoVectorsTwoDimensionsAndBatchesOfTwo() {

        val accumulator = SparseAccumulator(2, 2, 1, 2)

        accumulator.accumulate(intArrayOf(0), doubleArrayOf(1.1, 1.2))
        accumulator.accumulate(intArrayOf(1), doubleArrayOf(2.1, 2.2))

        check(accumulator, 2, intArrayOf(0, 1), doubleArrayOf(1.0, 1.0), arrayOf(doubleArrayOf(1.1, 1.2), doubleArrayOf(2.1, 2.2)))

    }

    @Test
    fun testTwoVectorsTwoDimensionsAndLengthsOfTwo() {

        val accumulator = SparseAccumulator(2, 1, 2, 2)

        accumulator.accumulate(intArrayOf(0, 1), doubleArrayOf(1.1, 1.2, 2.1, 2.2))

        check(accumulator, 2, intArrayOf(0, 1), doubleArrayOf(1.0, 1.0), arrayOf(doubleArrayOf(1.1, 1.2), doubleArrayOf(2.1, 2.2)))

    }

    @Test
    fun testTwoVectorsTwoDimensionsBatchesOfTwoAndLengthsOfTwo() {

        val accumulator = SparseAccumulator(2, 1, 2, 2)

        accumulator.accumulate(intArrayOf(0, 1), doubleArrayOf(1.1, 1.2, 2.1, 2.2))
        accumulator.accumulate(intArrayOf(0, 1), doubleArrayOf(1.3, 1.4, 2.3, 2.4))

        check(accumulator, 2, intArrayOf(0, 1), doubleArrayOf(2.0, 2.0), arrayOf(doubleArrayOf(2.4, 2.6), doubleArrayOf(4.4, 4.6)))

    }

    @Test
    fun testTwoVectorsTwoDimensionsBatchesOfTwoAndLengthsOfTwoReversed() {

        val accumulator = SparseAccumulator(2, 1, 2, 2)

        accumulator.accumulate(intArrayOf(0, 1), doubleArrayOf(1.1, 1.2, 2.1, 2.2))
        accumulator.accumulate(intArrayOf(1, 0), doubleArrayOf(2.3, 2.4, 1.3, 1.4))

        check(accumulator, 2, intArrayOf(0, 1), doubleArrayOf(2.0, 2.0), arrayOf(doubleArrayOf(2.4, 2.6), doubleArrayOf(4.4, 4.6)))

    }

    @Test
    fun testRepeatOnce() {

        val accumulator = SparseAccumulator(1, 1, 2, 1)

        accumulator.accumulate(intArrayOf(0, 0), doubleArrayOf(1.0, 2.0))

        check(accumulator, 1, intArrayOf(0), doubleArrayOf(1.0), arrayOf(doubleArrayOf(3.0)))

    }

    @Test
    fun testRepeatTwice() {

        val accumulator = SparseAccumulator(1, 2, 2, 1)

        accumulator.accumulate(intArrayOf(0, 0), doubleArrayOf(1.0, 2.0))
        accumulator.accumulate(intArrayOf(0, 0), doubleArrayOf(3.0, 4.0))

        check(accumulator, 1, intArrayOf(0), doubleArrayOf(2.0), arrayOf(doubleArrayOf(10.0)))

    }

    private fun check(
        accumulator: SparseAccumulator,
        expectedSize: Int,
        expectedIds: IntArray,
        expectedCounts: DoubleArray,
        expectedSums: Array<DoubleArray>) {

        val actualSize = accumulator.getSize()
        val actualIds = accumulator.getIds()
        val actualCounts = accumulator.getCounts()
        val actualSums = accumulator.getSums()

        assertEquals(expectedSize, actualSize)

        for (index in 0..actualSize - 1) {

            assertEquals(expectedIds[index], actualIds[index])
            assertEquals(expectedCounts[index], actualCounts[index])
            assertArrayEquals(actualSums[index], expectedSums[index], 0.001)

        }



    }

}