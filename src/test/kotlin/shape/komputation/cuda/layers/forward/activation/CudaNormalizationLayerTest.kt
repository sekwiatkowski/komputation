package shape.komputation.cuda.layers.forward.activation

import jcuda.Pointer
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

        test(input, 1, 1, expected)

    }

    @Test
    fun testForwardTwoRowsOneColumn() {

        val input = doubleArrayOf(1.0, 1.0)
        val expected = doubleArrayOf(0.5, 0.5)

        test(input, 2, 1, expected)

    }

    @Test
    fun testForwardTwoRowsTwoColumns() {

        val input = doubleArrayOf(1.0, 1.0, 1.0, 3.0)
        val expected = doubleArrayOf(0.5, 0.5, 0.25, 0.75)

        test(input, 2, 2, expected)

    }

    private fun test(input: DoubleArray, numberRows : Int, numberColumns : Int, expected: DoubleArray) {

        val context = setUpCudaContext()

        val blockSize = Math.pow(2.0, Math.ceil(Math.log(numberRows.toDouble()) / Math.log(2.0))).toInt()

        val normalizationKernel = context.kernelFactory.normalizationKernel(blockSize)
        val layer = CudaNormalizationLayer(null, normalizationKernel, numberRows, numberColumns)
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

}