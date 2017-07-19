package shape.komputation.cuda.layers.forward.activation

import jcuda.Pointer
import jcuda.jcublas.cublasHandle
import jcuda.runtime.JCuda.cudaFree
import org.junit.jupiter.api.Assertions.assertArrayEquals
import org.junit.jupiter.api.Test
import shape.komputation.cpu.functions.activation.backwardColumnWiseSoftmax
import shape.komputation.cpu.functions.activation.columnWiseSoftmax
import shape.komputation.cuda.getVector
import shape.komputation.cuda.setUpCudaContext
import shape.komputation.cuda.setVector
import shape.komputation.layers.forward.activation.softmaxLayer

class CudaSoftmaxLayerTest {

    @Test
    fun testForwardOneRowOneColumn() {

        val input = doubleArrayOf(0.0)
        val numberRows = 1
        val numberColumns = 1

        testForward(numberRows, numberColumns, input, doubleArrayOf(1.0))

    }

    @Test
    fun testForwardTwoRowsOneColumn1() {

        val input = doubleArrayOf(0.0, 0.0)
        val numberRows = 2
        val numberColumns = 1

        val expected = doubleArrayOf(0.5, 0.5)

        testForward(numberRows, numberColumns, input, expected)

    }

    @Test
    fun testForwardTwoRowsOneColumn2() {

        val input = doubleArrayOf(0.0, 1.0)
        val numberRows = 2
        val numberColumns = 1

        val expected = doubleArrayOf(0.268941421, 0.731058579)

        testForward(numberRows, numberColumns, input, expected)

    }

    private fun testForward(numberRows: Int, numberColumns: Int, input: DoubleArray, expected: DoubleArray) {

        val actual = forward(numberRows, numberColumns, input)

        assertArrayEquals(expected, actual, 0.001)

    }

    private fun forward(numberRows: Int, numberColumns: Int, input: DoubleArray): DoubleArray {

        val numberEntries = numberRows * numberColumns

        val cudaContext = setUpCudaContext()

        val softmaxLayer = softmaxLayer(numberRows, numberColumns).buildForCuda(cudaContext, cublasHandle())

        softmaxLayer.acquire()

        val deviceInput = Pointer()
        setVector(input, numberEntries, deviceInput)

        val deviceResult = softmaxLayer.forward(deviceInput)
        val actual = getVector(deviceResult, numberEntries)

        cudaFree(deviceInput)

        softmaxLayer.release()

        cudaContext.destroy()

        return actual

    }

    @Test
    fun testBackwardOneRowOneColumn() {

        val input = doubleArrayOf(1.0)
        val chain = doubleArrayOf(1.0)
        val numberRows = 1
        val numberColumns = 1

        testBackward(numberRows, numberColumns, input, chain)

    }

    @Test
    fun testBackwardTwoRowsOneColumn1() {

        val input = doubleArrayOf(1.0, 1.0)
        val chain = doubleArrayOf(1.0, 1.0)
        val numberRows = 2
        val numberColumns = 1

        testBackward(numberRows, numberColumns, input, chain)

    }

    @Test
    fun testBackwardTwoRowsOneColumn2() {

        val input = doubleArrayOf(0.0, 1.0)
        val chain = doubleArrayOf(1.0, 1.0)
        val numberRows = 2
        val numberColumns = 1

        testBackward(numberRows, numberColumns, input, chain)

    }

    @Test
    fun testBackwardOneRowTwoColumns() {

        val input = doubleArrayOf(1.0, 2.0)
        val chain = doubleArrayOf(1.0, 2.0)
        val numberRows = 1
        val numberColumns = 2

        testBackward(numberRows, numberColumns, input, chain)

    }


    private fun testBackward(numberRows: Int, numberColumns: Int, input: DoubleArray, chain : DoubleArray) {

        val forwardEntries = columnWiseSoftmax(input, numberRows, numberColumns)
        val expected = backwardColumnWiseSoftmax(numberRows, numberColumns, forwardEntries, chain)

        val actual = backward(numberRows, numberColumns, input, chain)

        assertArrayEquals(expected, actual, 0.001)

    }

    private fun backward(numberRows: Int, numberColumns: Int, input: DoubleArray, chain: DoubleArray): DoubleArray {

        val numberEntries = numberRows * numberColumns

        val cudaContext = setUpCudaContext()

        val softmaxLayer = softmaxLayer(numberRows, numberColumns).buildForCuda(cudaContext, cublasHandle())

        softmaxLayer.acquire()

        val deviceInput = Pointer()
        setVector(input, numberEntries, deviceInput)

        val deviceChain = Pointer()
        setVector(chain, numberEntries, deviceChain)

        softmaxLayer.forward(deviceInput)
        val deviceBackwardResult = softmaxLayer.backward(deviceChain)
        val actual = getVector(deviceBackwardResult, numberEntries)

        cudaFree(deviceInput)
        cudaFree(deviceChain)

        softmaxLayer.release()

        cudaContext.destroy()

        return actual

    }


}