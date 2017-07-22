package shape.komputation.cuda.layers.forward.activation

import jcuda.jcublas.cublasHandle
import org.junit.jupiter.api.Test
import shape.komputation.cpu.functions.activation.sigmoid
import shape.komputation.cuda.CudaContext
import shape.komputation.layers.forward.activation.sigmoidLayer

class CudaSigmoidLayerTest : BaseCudaEntrywiseActivationLayerTest() {

    override fun createLayer(context: CudaContext, numberEntries: Int)  =

        sigmoidLayer(numberEntries).buildForCuda(context, cublasHandle())

    @Test
    fun testForwardOneDimension() {

        val input = floatArrayOf(0.0f)
        val expected = floatArrayOf(0.5f)

        testForward(input, expected)

    }

    @Test
    fun testForwardTwoDimensions() {

        val input = floatArrayOf(0.0f, 1.0f)
        val expected = floatArrayOf(0.5f, 0.731058579f)

        testForward(input, expected)

    }


    @Test
    fun testBackwardOneDimension() {

        val input = floatArrayOf(0.0f)
        val chain = floatArrayOf(1.0f)
        val expected = floatArrayOf(0.25f)

        testBackward(input, chain, expected)

    }

    @Test
    fun testBackwardTwoDimensions() {

        val input = floatArrayOf(0.0f, 1.0f)
        val chain = floatArrayOf(1.0f, 2.0f)
        val expected = floatArrayOf(1.0f * 0.5f * (1.0f - 0.5f), 2.0f * sigmoid(1.0f) * (1.0f - sigmoid(1.0f)))

        testBackward(input, chain, expected)

    }

}