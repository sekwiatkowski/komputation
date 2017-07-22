package shape.komputation.cuda.layers.forward.activation

import jcuda.jcublas.cublasHandle
import org.junit.jupiter.api.Test
import shape.komputation.cuda.CudaContext
import shape.komputation.layers.forward.activation.tanhLayer
import shape.komputation.matrix.FloatMath

class CudaTanhLayerTest : BaseCudaEntrywiseActivationLayerTest() {

    override fun createLayer(context: CudaContext, numberEntries: Int) =

        tanhLayer(numberEntries).buildForCuda(context, cublasHandle())

    @Test
    fun testForwardOneDimension1() {

        val input = floatArrayOf(0.0f)
        val expected = floatArrayOf(0.0f)

        testForward(input, expected)

    }

    @Test
    fun testForwardOneDimension2() {

        val input = floatArrayOf(1.0f)
        val expected = floatArrayOf(0.761594156f)

        testForward(input, expected)

    }


    @Test
    fun testForwardOneDimension3() {

        val input = floatArrayOf(-1.0f)
        val expected = floatArrayOf(-0.761594156f)

        testForward(input, expected)

    }

    @Test
    fun testForwardTwoDimensions() {

        val input = floatArrayOf(-2.0f, 3.0f)
        val expected = floatArrayOf(-0.96402758f, 0.995054754f)

        testForward(input, expected)

    }

    @Test
    fun testBackwardOneDimension() {

        val input = floatArrayOf(1.0f)
        val chain = floatArrayOf(1.0f)
        val expected = floatArrayOf(1.0f * (1.0f - FloatMath.pow(FloatMath.tanh(1.0f), 2.0f)))

        testBackward(input, chain, expected)

    }

}