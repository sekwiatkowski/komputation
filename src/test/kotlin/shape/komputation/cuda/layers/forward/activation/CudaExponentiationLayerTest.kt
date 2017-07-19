package shape.komputation.cuda.layers.forward.activation

import jcuda.Pointer
import jcuda.jcublas.cublasHandle
import jcuda.runtime.JCuda.cudaFree
import org.junit.jupiter.api.Assertions.assertArrayEquals
import org.junit.jupiter.api.Test
import shape.komputation.cuda.getVector
import shape.komputation.cuda.setUpCudaContext
import shape.komputation.cuda.setVector
import shape.komputation.layers.forward.activation.exponentiationLayer

class CudaExponentiationLayerTest {

    @Test
    fun testForwardOneRowOneColumn() {

        val input = doubleArrayOf(1.0)
        val numberRows = 1
        val numberColumns = 1

        testForward(numberRows, numberColumns, input, doubleArrayOf(Math.exp(1.0)))

    }

    @Test
    fun testForwardTwoRowsOneColumn() {

        val input = doubleArrayOf(0.0, 1.0)
        val numberRows = 2
        val numberColumns = 1

        testForward(numberRows, numberColumns, input, doubleArrayOf(1.0, Math.exp(1.0)))

    }

    private fun testForward(numberRows: Int, numberColumns: Int, input: DoubleArray, expected: DoubleArray) {

        val actual = forward(numberRows, numberColumns, input)

        assertArrayEquals(expected, actual, 0.001)

    }

    private fun forward(numberRows: Int, numberColumns: Int, input: DoubleArray): DoubleArray {

        val numberEntries = numberRows * numberColumns

        val cudaContext = setUpCudaContext()

        val layer = exponentiationLayer(numberRows, numberColumns).buildForCuda(cudaContext, cublasHandle())

        layer.acquire()

        val deviceInput = Pointer()
        setVector(input, numberEntries, deviceInput)

        val deviceResult = layer.forward(deviceInput)
        val actual = getVector(deviceResult, numberEntries)

        cudaFree(deviceInput)

        layer.release()

        cudaContext.destroy()

        return actual

    }

}