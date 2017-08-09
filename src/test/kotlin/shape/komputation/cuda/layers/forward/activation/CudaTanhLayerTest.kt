package shape.komputation.cuda.layers.forward.activation

import jcuda.jcublas.cublasHandle
import org.junit.jupiter.api.Test
import shape.komputation.cuda.CudaContext
import shape.komputation.layers.forward.activation.tanhLayer
import shape.komputation.matrix.FloatMath

class CudaTanhLayerTest : BaseCudaEntrywiseActivationLayerTest() {

    override fun createLayer(context: CudaContext, numberRows: Int) =

        tanhLayer(numberRows).buildForCuda(context, cublasHandle())

    @Test
    fun testForwardOneOfTwoInstancesOneDimensional() {

        val input = floatArrayOf(1.0f, 0.0f)
        val expected = floatArrayOf(FloatMath.tanh(1.0f), 0.0f)

        testForward(input, 1, 2, expected)

    }

    @Test
    fun testForwardOneOfTwoInstancesTwoDimensional() {

        val input = floatArrayOf(1.0f, 2.0f, 0.0f, 0.0f)
        val expected = floatArrayOf(FloatMath.tanh(1.0f), FloatMath.tanh(2.0f), 0.0f, 0.0f)

        testForward(input, 1, 2, expected)

    }

    @Test
    fun testBackwardOneDimension() {

        val input = floatArrayOf(1.0f, 0.0f)
        val chain = floatArrayOf(1.0f, 0.0f)
        val expected = floatArrayOf(1.0f * (1.0f - FloatMath.pow(FloatMath.tanh(1.0f), 2.0f)), 0.0f)

        testBackward(input, chain, 1, 2, expected)

    }

}