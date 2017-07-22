package shape.komputation.cuda.layers.forward.activation

import jcuda.jcublas.cublasHandle
import org.junit.jupiter.api.Test
import shape.komputation.cuda.CudaContext
import shape.komputation.layers.forward.activation.exponentiationLayer
import shape.komputation.matrix.FloatMath

class CudaExponentiationLayerTest : BaseCudaEntrywiseActivationLayerTest() {

    override fun createLayer(context: CudaContext, numberEntries: Int) =

        exponentiationLayer(numberEntries).buildForCuda(context, cublasHandle())

    @Test
    fun testForwardOneDimension() {

        val input = floatArrayOf(1.0f)
        val expected = floatArrayOf(FloatMath.exp(1.0f))

        testForward(input, expected)

    }

    @Test
    fun testForwardTwoDimensions() {

        val input = floatArrayOf(0.0f, 1.0f)
        val expected = floatArrayOf(1.0f, FloatMath.exp(1.0f))

        testForward(input, expected)

    }


    @Test
    fun testBackwardOneDimension() {

        val input = floatArrayOf(1.0f)
        val chain = floatArrayOf(2.0f)
        val expected = floatArrayOf(2.0f * FloatMath.exp(1.0f))

        testBackward(input, chain, expected)

    }

    @Test
    fun testBackwardTwoDimensions() {

        val input = floatArrayOf(0.0f, 1.0f)
        val chain = floatArrayOf(2.0f, 3.0f)
        val expected = floatArrayOf(2.0f, 3.0f * FloatMath.exp(1.0f))

        testBackward(input, chain, expected)

    }


}