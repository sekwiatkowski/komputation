package shape.komputation.cuda.layers.forward.maxpooling

import jcuda.Pointer
import jcuda.jcublas.cublasHandle
import org.junit.jupiter.api.Assertions.assertArrayEquals
import org.junit.jupiter.api.Test
import shape.komputation.cuda.getFloatArray
import shape.komputation.cuda.setFloatArray
import shape.komputation.cuda.setUpCudaContext
import shape.komputation.layers.forward.convolution.maxPoolingLayer

class CudaMaxPoolingLayerTest {

    @Test
    fun testOneRowOneColumn() {

        val input = floatArrayOf(1.0f)
        val expected = floatArrayOf(1.0f)

        test(1, 1, input, expected)

    }

    @Test
    fun testOneRowTwoColumns() {

        val input = floatArrayOf(1.0f, 2.0f)
        val expected = floatArrayOf(2.0f)

        test(1, 2, input, expected)

    }

    @Test
    fun testOneRowTwoColumnsReversed() {

        val input = floatArrayOf(2.0f, 1.0f)
        val expected = floatArrayOf(2.0f)

        test(1, 2, input, expected)

    }

    @Test
    fun testOneRowThreeColumns() {

        val input = floatArrayOf(1.0f, 3.0f, 2.0f)
        val expected = floatArrayOf(3.0f)

        test(1, 3, input, expected)

    }

    @Test
    fun testOneRowThirtyThree() {

        val input = FloatArray(33) { index -> (index+1).toFloat() }
        val expected = floatArrayOf(33.0f)

        test(1, 33, input, expected)

    }

    @Test
    fun testTwoRowsOneColumn() {

        val input = floatArrayOf(1.0f, 2.0f)
        val expected = floatArrayOf(1.0f, 2.0f)

        test(2, 1, input, expected)

    }

    @Test
    fun testTwoRowsTwoColumns() {

        /*
            1.0 3.0
            2.0 -4.0
        */
        val input = floatArrayOf(1.0f, 2.0f, 3.0f, -4.0f)
        val expected = floatArrayOf(3.0f, 2.0f)

        test(2, 2, input, expected)

    }

    private fun test(numberRows : Int, numberColumns : Int, input : FloatArray, expected : FloatArray) {

        val cudaContext = setUpCudaContext()

        val maxPoolingLayer = maxPoolingLayer(numberRows, numberColumns).buildForCuda(cudaContext, cublasHandle())

        val batchSize = 1
        maxPoolingLayer.acquire(batchSize)

        val deviceInput = Pointer()
        setFloatArray(input, numberRows * numberColumns, deviceInput)

        val deviceResult = maxPoolingLayer.forward(deviceInput, batchSize, false)

        val actual = getFloatArray(deviceResult, numberRows)

        cudaContext.destroy()

        assertArrayEquals(expected, actual, 0.01f)

    }



}