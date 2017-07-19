package shape.komputation.cuda.layers.forward

import jcuda.Pointer
import jcuda.jcublas.cublasHandle
import jcuda.runtime.JCuda.cudaFree
import org.junit.jupiter.api.Assertions.assertArrayEquals
import org.junit.jupiter.api.Test
import shape.komputation.cuda.getVector
import shape.komputation.cuda.setUpCudaContext
import shape.komputation.cuda.setVector
import shape.komputation.layers.forward.normalizationLayer

class CudaNormalizationLayerTest {

    @Test
    fun testForwardOneRowOneColumn() {

        val input = doubleArrayOf(1.0)

        this.forward(input, 1, 1)

    }

    @Test
    fun testForwardTwoRowsOneColumn() {

        val input = doubleArrayOf(1.0, 1.0)
        val expected = doubleArrayOf(0.5, 0.5)

        testForward(input, 2, 1, expected)

    }

    @Test
    fun testForwardTwoRowsTwoColumns() {

        val input = doubleArrayOf(1.0, 1.0, 1.0, 3.0)
        val expected = doubleArrayOf(0.5, 0.5, 0.25, 0.75)

        testForward(input, 2, 2, expected)

    }

    @Test
    fun testBackwardOneRowOneColumn() {

        val input = doubleArrayOf(1.0)
        val chain = doubleArrayOf(1.0)

        this.backward(1, 1, input, chain)

    }

    @Test
    fun testBackwardTwoRowsOneColumn1() {

        val input = doubleArrayOf(1.0, 1.0)
        val chain = doubleArrayOf(1.0, 1.0)
        val expected = doubleArrayOf(0.0, 0.0)

        testBackward(2, 1, input, chain, expected)

    }

    @Test
    fun testBackwardTwoRowsOneColumn2() {

        val input = doubleArrayOf(1.0, 2.0)
        val chain = doubleArrayOf(1.0, 1.0)
        val expected = doubleArrayOf(0.0, 0.0)

        testBackward(2, 1, input, chain, expected)

    }

    @Test
    fun testBackwardTwoRowsOneColumn3() {

        val input = doubleArrayOf(1.0, 1.0)
        val chain = doubleArrayOf(2.0, 1.0)
        val sum = 2.0
        val expectedFirst = (2.0 * 1.0 - 1.0 * 1.0) / Math.pow(sum, 2.0)
        val expectedSecond = (-2.0 * 1.0 + 1.0 * 1.0) / Math.pow(sum, 2.0)
        val expected = doubleArrayOf(expectedFirst, expectedSecond)

        testBackward(2, 1, input, chain, expected)

    }

    @Test
    fun testBackwardOneRowTwoColumns() {

        val input = doubleArrayOf(1.0, 2.0)
        val chain = doubleArrayOf(1.0, 2.0)
        val numberRows = 1
        val numberColumns = 2

        val expected = doubleArrayOf(0.0, 0.0)

        testBackward(numberRows, numberColumns, input, chain, expected)

    }

    @Test
    fun testBackwardTwoRowsTwoColumns() {

        val input = doubleArrayOf(0.0, 1.0, 2.0, 3.0)
        val chain = doubleArrayOf(4.0, 5.0, 6.0, 7.0)
        val numberRows = 2
        val numberColumns = 2

        val firstSum = 0.0+1.0
        val firstSquaredSum = Math.pow(firstSum, 2.0)
        val secondSum = 2.0+3.0
        val secondSquaredSum = Math.pow(secondSum, 2.0)

        val expected = doubleArrayOf(
            (4.0 * 1.0 - 5.0 * 1.0) / firstSquaredSum,
            (-4.0 * 0.0 + 5.0 * 0.0) / firstSquaredSum,
            (6.0 * 3.0 - 7.0 * 3.0) / secondSquaredSum,
            (-6.0 * 2.0 + 7.0 * 2.0) / secondSquaredSum)

        testBackward(numberRows, numberColumns, input, chain, expected)

    }

    private fun testForward(input: DoubleArray, numberRows : Int, numberColumns : Int, expected: DoubleArray) {

        val actual = forward(input, numberRows, numberColumns)

        assertArrayEquals(expected, actual, 0.001)

    }

    private fun forward(input: DoubleArray, numberRows: Int, numberColumns: Int): DoubleArray {

        val context = setUpCudaContext()

        val layer = normalizationLayer(numberRows, numberColumns).buildForCuda(context, cublasHandle())
        layer.acquire()

        val deviceInput = Pointer()
        val numberEntries = numberRows * numberColumns
        setVector(input, numberEntries, deviceInput)

        val deviceResult = layer.forward(deviceInput)
        val actual = getVector(deviceResult, numberEntries)

        cudaFree(deviceInput)

        layer.release()

        context.destroy()

        return actual

    }


    private fun testBackward(numberRows : Int, numberColumns : Int, input : DoubleArray, chain: DoubleArray, expected: DoubleArray) {

        val actual = backward(numberRows, numberColumns, input, chain)

        assertArrayEquals(expected, actual, 0.001)

    }

    private fun backward(numberRows: Int, numberColumns: Int, input: DoubleArray, chain: DoubleArray): DoubleArray {

        val context = setUpCudaContext()

        val numberEntries = numberRows * numberColumns

        val deviceInput = Pointer()
        setVector(input, numberEntries, deviceInput)

        val deviceChain = Pointer()
        setVector(chain, numberEntries, deviceChain)

        val layer = normalizationLayer(numberRows, numberColumns).buildForCuda(context, cublasHandle())
        layer.acquire()

        layer.forward(deviceInput)
        val deviceBackwardResult = layer.backward(deviceChain)
        val actual = getVector(deviceBackwardResult, numberEntries)

        layer.release()

        cudaFree(deviceInput)
        cudaFree(deviceChain)

        context.destroy()

        return actual

    }

}