package shape.komputation.cuda.layers.forward.activation

import jcuda.Pointer
import jcuda.jcublas.cublasHandle
import jcuda.runtime.JCuda.cudaFree
import org.junit.jupiter.api.Assertions.assertArrayEquals
import org.junit.jupiter.api.Test
import shape.komputation.cuda.getVector
import shape.komputation.cuda.setUpCudaContext
import shape.komputation.cuda.setVector
import shape.komputation.layers.forward.activation.softmaxLayer
import shape.komputation.matrix.FloatMatrix

class CudaSoftmaxLayerTest {

    @Test
    fun testForwardOneRowOneColumn() {

        val input = floatArrayOf(0.0f)
        val numberRows = 1
        val numberColumns = 1

        testForward(numberRows, numberColumns, input, floatArrayOf(1.0f))

    }

    @Test
    fun testForwardTwoRowsOneColumn1() {

        val input = floatArrayOf(0.0f, 0.0f)
        val numberRows = 2
        val numberColumns = 1

        val expected = floatArrayOf(0.5f, 0.5f)

        testForward(numberRows, numberColumns, input, expected)

    }

    @Test
    fun testForwardTwoRowsOneColumn2() {

        val input = floatArrayOf(0.0f, 1.0f)
        val numberRows = 2
        val numberColumns = 1

        val expected = floatArrayOf(0.268941421f, 0.731058579f)

        testForward(numberRows, numberColumns, input, expected)

    }

    private fun testForward(numberRows: Int, numberColumns: Int, input: FloatArray, expected: FloatArray) {

        val actual = forward(numberRows, numberColumns, input)

        assertArrayEquals(expected, actual, 0.001f)

    }

    private fun forward(numberRows: Int, numberColumns: Int, input: FloatArray): FloatArray {

        val numberEntries = numberRows * numberColumns

        val cudaContext = setUpCudaContext()

        val softmaxLayer = softmaxLayer(numberRows, numberColumns).buildForCuda(cudaContext, cublasHandle())

        softmaxLayer.acquire()

        val deviceInput = Pointer()
        setVector(input, numberEntries, deviceInput)

        val deviceResult = softmaxLayer.forward(deviceInput, true)
        val actual = getVector(deviceResult, numberEntries)

        cudaFree(deviceInput)

        softmaxLayer.release()

        cudaContext.destroy()

        return actual

    }

    @Test
    fun testBackwardOneRowOneColumn() {

        val input = floatArrayOf(1.0f)
        val chain = floatArrayOf(1.0f)
        val numberRows = 1
        val numberColumns = 1

        testBackward(numberRows, numberColumns, input, chain)

    }

    @Test
    fun testBackwardTwoRowsOneColumn1() {

        val input = floatArrayOf(1.0f, 1.0f)
        val chain = floatArrayOf(1.0f, 1.0f)
        val numberRows = 2
        val numberColumns = 1

        testBackward(numberRows, numberColumns, input, chain)

    }

    @Test
    fun testBackwardTwoRowsOneColumn2() {

        val input = floatArrayOf(0.0f, 1.0f)
        val chain = floatArrayOf(1.0f, 1.0f)
        val numberRows = 2
        val numberColumns = 1

        testBackward(numberRows, numberColumns, input, chain)

    }

    @Test
    fun testBackwardOneRowTwoColumns() {

        val input = floatArrayOf(1.0f, 2.0f)
        val chain = floatArrayOf(1.0f, 2.0f)
        val numberRows = 1
        val numberColumns = 2

        testBackward(numberRows, numberColumns, input, chain)

    }


    private fun testBackward(numberRows: Int, numberColumns: Int, input: FloatArray, chain : FloatArray) {

        val numberEntries = numberRows * numberColumns

        val cudaContext = setUpCudaContext()

        val softmaxLayer = softmaxLayer(numberRows, numberColumns)

        val cpuSoftmaxLayer = softmaxLayer.buildForCpu()
        cpuSoftmaxLayer.forward(FloatMatrix(numberRows, numberColumns, input), true)
        val cpuBackward = cpuSoftmaxLayer.backward(FloatMatrix(numberRows, numberColumns, chain))
        val expected = cpuBackward.entries

        val cudaSoftmaxLayer = softmaxLayer.buildForCuda(cudaContext, cublasHandle())

        cudaSoftmaxLayer.acquire()

        val deviceInput = Pointer()
        setVector(input, numberEntries, deviceInput)

        val deviceChain = Pointer()
        setVector(chain, numberEntries, deviceChain)

        cudaSoftmaxLayer.forward(deviceInput, true)
        val deviceBackwardResult = cudaSoftmaxLayer.backward(deviceChain)
        val actual = getVector(deviceBackwardResult, numberEntries)

        cudaFree(deviceInput)
        cudaFree(deviceChain)

        cudaSoftmaxLayer.release()

        cudaContext.destroy()

        assertArrayEquals(expected, actual, 0.001f)

    }


}