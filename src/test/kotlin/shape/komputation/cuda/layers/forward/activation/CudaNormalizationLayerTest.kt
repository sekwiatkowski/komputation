package shape.komputation.cuda.layers.forward.activation

import jcuda.Pointer
import jcuda.Sizeof
import jcuda.runtime.JCuda.cudaFree
import org.junit.jupiter.api.Assertions.assertArrayEquals
import org.junit.jupiter.api.Test
import shape.komputation.cuda.getVector
import shape.komputation.cuda.setUpCudaContext
import shape.komputation.cuda.setVector

class CudaNormalizationLayerTest {

    @Test
    fun testForwardOneRowOneColumn() {

        val input = doubleArrayOf(1.0)
        val expected = doubleArrayOf(1.0)

        testForward(input, 1, 1, expected)

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
        val sums = doubleArrayOf(1.0)
        val expected = doubleArrayOf(0.0)

        testBackward(input, 1, 1, sums, expected)

    }

    @Test
    fun testBackwardTwoRowsOneColumn() {

        val input = doubleArrayOf(1.0, 2.0)
        val sums = doubleArrayOf(3.0)
        val expected = doubleArrayOf(2.0/9.0, 1.0/9.0)

        testBackward(input, 2, 1, sums, expected)

    }

    @Test
    fun testBackwardTwoRowsTwoColumns() {

        val input = doubleArrayOf(1.0, 2.0, 3.0, 4.0)
        val sums = doubleArrayOf(3.0, 7.0)
        val expected = doubleArrayOf(2.0/9.0, 1.0/9.0, 4.0/49.0, 3.0/49.0)

        testBackward(input, 2, 2, sums, expected)

    }


    private fun testForward(input: DoubleArray, numberRows : Int, numberColumns : Int, expected: DoubleArray) {

        val context = setUpCudaContext()

        val blockSize = Math.pow(2.0, Math.ceil(Math.log(numberRows.toDouble()) / Math.log(2.0))).toInt()

        val layer = CudaNormalizationLayer(null, context.kernelFactory.forwardNormalizationKernel(blockSize), context.kernelFactory.backwardNormalizationKernel(), numberRows, numberColumns)
        layer.acquire()

        val deviceInput = Pointer()
        val numberEntries = numberRows * numberColumns
        setVector(input, numberEntries, deviceInput)

        val deviceResult = layer.forward(deviceInput)
        val actual = getVector(deviceResult, numberEntries)

        cudaFree(deviceInput)

        layer.release()

        context.destroy()

        assertArrayEquals(expected, actual, 0.001)

    }


    private fun testBackward(input: DoubleArray, numberRows : Int, numberColumns : Int, sums : DoubleArray, expected: DoubleArray) {

        val context = setUpCudaContext()

        val kernel = context.kernelFactory.backwardNormalizationKernel()
        kernel.acquire()

        val numberEntries = numberRows * numberColumns

        val deviceInput = Pointer()
        setVector(input, numberEntries, deviceInput)

        val deviceResult = Pointer()
        setVector(input, numberEntries, deviceResult)

        val deviceSums = Pointer()
        setVector(sums, sums.size, deviceSums)

        kernel.launch(
            Pointer.to(
                Pointer.to(deviceInput),
                Pointer.to(deviceSums),
                Pointer.to(deviceResult)
            ),
            numberColumns,
            numberRows,
            Sizeof.DOUBLE
        )
        val actual = getVector(deviceResult, numberEntries)

        kernel.release()

        cudaFree(deviceInput)
        cudaFree(deviceSums)
        cudaFree(deviceResult)

        context.destroy()

        assertArrayEquals(expected, actual, 0.001)

    }

}