package shape.komputation.cuda.layers.forward.activation

import jcuda.jcublas.cublasHandle
import org.junit.jupiter.api.Test
import shape.komputation.cuda.CudaContext
import shape.komputation.layers.forward.activation.reluLayer

class CudaReluLayerTest : BaseCudaEntrywiseActivationLayerTest() {

    override fun createLayer(context: CudaContext, numberEntries: Int) =

        reluLayer(numberEntries).buildForCuda(context, cublasHandle())

    @Test
    fun testForwardOneDimension1() {

        val input = floatArrayOf(1.0f)
        val expected = floatArrayOf(1.0f)

        testForward(input, expected)

    }

    @Test
    fun testForwardOneDimension2() {

        val input = floatArrayOf(-1.0f)
        val expected = floatArrayOf(0.0f)

        testForward(input, expected)

    }

    @Test
    fun testForwardOneDimension3() {

        val input = floatArrayOf(0.0f)
        val expected = floatArrayOf(0.0f)

        testForward(input, expected)

    }


    @Test
    fun testForwardTwoDimensions() {

        val input = floatArrayOf(-2.0f, 2.0f)
        val expected = floatArrayOf(0.0f, 2.0f)

        testForward(input, expected)


    }

    @Test
    fun testBackwardOneDimension1() {

        val input = floatArrayOf(1.0f)
        val chain = floatArrayOf(2.0f)
        val expected = floatArrayOf(2.0f)

        testBackward(input, chain, expected)

    }

    @Test
    fun testBackwardOneDimension2() {

        val input = floatArrayOf(-1.0f)
        val chain = floatArrayOf(2.0f)
        val expected = floatArrayOf(0.0f)

        testBackward(input, chain, expected)

    }

    @Test
    fun testBackwardOneDimension3() {

        val input = floatArrayOf(0.0f)
        val chain = floatArrayOf(2.0f)
        val expected = floatArrayOf(0.0f)

        testBackward(input, chain, expected)

    }


    @Test
    fun testBackwardTwoDimensions() {

        val input = floatArrayOf(-1.0f, 1.0f)
        val chain = floatArrayOf(1.0f, 2.0f)
        val expected = floatArrayOf(0.0f, 2.0f)

        testBackward(input, chain, expected)

    }


}