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
import shape.komputation.matrix.FloatMath

class CudaNormalizationLayerTest {

    @Test
    fun testForwardOneRowOneColumn() {

        val input = floatArrayOf(1.0f)

        this.forward(input, 1, 1)

    }

    @Test
    fun testForwardTwoRowsOneColumn() {

        val input = floatArrayOf(1.0f, 1.0f)
        val expected = floatArrayOf(0.5f, 0.5f)

        testForward(input, 2, 1, expected)

    }

    @Test
    fun testForwardTwoRowsTwoColumns() {

        val input = floatArrayOf(1.0f, 1.0f, 1.0f, 3.0f)
        val expected = floatArrayOf(0.5f, 0.5f, 0.25f, 0.75f)

        testForward(input, 2, 2, expected)

    }

    @Test
    fun testBackwardOneRowOneColumn() {

        val input = floatArrayOf(1.0f)
        val chain = floatArrayOf(1.0f)

        this.backward(1, 1, input, chain)

    }

    @Test
    fun testBackwardTwoRowsOneColumn1() {

        val input = floatArrayOf(1.0f, 1.0f)
        val chain = floatArrayOf(1.0f, 1.0f)
        val expected = floatArrayOf(0.0f, 0.0f)

        testBackward(2, 1, input, chain, expected)

    }

    @Test
    fun testBackwardTwoRowsOneColumn2() {

        val input = floatArrayOf(1.0f, 2.0f)
        val chain = floatArrayOf(1.0f, 1.0f)
        val expected = floatArrayOf(0.0f, 0.0f)

        testBackward(2, 1, input, chain, expected)

    }

    @Test
    fun testBackwardTwoRowsOneColumn3() {

        val input = floatArrayOf(1.0f, 1.0f)
        val chain = floatArrayOf(2.0f, 1.0f)
        val sum = 2.0f
        val expectedFirst = (2.0f * 1.0f - 1.0f * 1.0f) / FloatMath.pow(sum, 2.0f)
        val expectedSecond = (-2.0f * 1.0f + 1.0f * 1.0f) / FloatMath.pow(sum, 2.0f)
        val expected = floatArrayOf(expectedFirst, expectedSecond)

        testBackward(2, 1, input, chain, expected)

    }

    @Test
    fun testBackwardOneRowTwoColumns() {

        val input = floatArrayOf(1.0f, 2.0f)
        val chain = floatArrayOf(1.0f, 2.0f)
        val numberRows = 1
        val numberColumns = 2

        val expected = floatArrayOf(0.0f, 0.0f)

        testBackward(numberRows, numberColumns, input, chain, expected)

    }

    @Test
    fun testBackwardTwoRowsTwoColumns() {

        val input = floatArrayOf(0.0f, 1.0f, 2.0f, 3.0f)
        val chain = floatArrayOf(4.0f, 5.0f, 6.0f, 7.0f)
        val numberRows = 2
        val numberColumns = 2

        val firstSum = 0.0f+1.0f
        val firstSquaredSum = firstSum * firstSum
        val secondSum = 2.0f+3.0f
        val secondSquaredSum = secondSum * secondSum

        val expected = floatArrayOf(
            (4.0f * 1.0f - 5.0f * 1.0f) / firstSquaredSum,
            (-4.0f * 0.0f + 5.0f * 0.0f) / firstSquaredSum,
            (6.0f * 3.0f - 7.0f * 3.0f) / secondSquaredSum,
            (-6.0f * 2.0f + 7.0f * 2.0f) / secondSquaredSum)

        testBackward(numberRows, numberColumns, input, chain, expected)

    }

    private fun testForward(input: FloatArray, numberRows : Int, numberColumns : Int, expected: FloatArray) {

        val actual = forward(input, numberRows, numberColumns)

        assertArrayEquals(expected, actual, 0.001f)

    }

    private fun forward(input: FloatArray, numberRows: Int, numberColumns: Int): FloatArray {

        val context = setUpCudaContext()

        val layer = normalizationLayer(numberRows, numberColumns).buildForCuda(context, cublasHandle())
        layer.acquire()

        val deviceInput = Pointer()
        val numberEntries = numberRows * numberColumns
        setVector(input, numberEntries, deviceInput)

        val deviceResult = layer.forward(deviceInput, false)
        val actual = getVector(deviceResult, numberEntries)

        cudaFree(deviceInput)

        layer.release()

        context.destroy()

        return actual

    }


    private fun testBackward(numberRows : Int, numberColumns : Int, input : FloatArray, chain: FloatArray, expected: FloatArray) {

        val actual = backward(numberRows, numberColumns, input, chain)

        assertArrayEquals(expected, actual, 0.001f)

    }

    private fun backward(numberRows: Int, numberColumns: Int, input: FloatArray, chain: FloatArray): FloatArray {

        val context = setUpCudaContext()

        val numberEntries = numberRows * numberColumns

        val deviceInput = Pointer()
        setVector(input, numberEntries, deviceInput)

        val deviceChain = Pointer()
        setVector(chain, numberEntries, deviceChain)

        val layer = normalizationLayer(numberRows, numberColumns).buildForCuda(context, cublasHandle())
        layer.acquire()

        layer.forward(deviceInput, true)
        val deviceBackwardResult = layer.backward(deviceChain)
        val actual = getVector(deviceBackwardResult, numberEntries)

        layer.release()

        cudaFree(deviceInput)
        cudaFree(deviceChain)

        context.destroy()

        return actual

    }

}