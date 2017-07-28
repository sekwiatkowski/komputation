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
    fun testForwardOneOfTwoInstancesOneDimensional() {

        val input = floatArrayOf(1.0f, 0.0f)
        val expected = floatArrayOf(sigmoid(1.0f), 0.0f)

        testForward(input, 1, 2, expected)

    }

    @Test
    fun testForwardOneOfTwoInstancesTwoDimensional() {

        val input = floatArrayOf(1.0f, 2.0f, 0.0f, 0.0f)
        val expected = floatArrayOf(sigmoid(1.0f), sigmoid(2.0f), 0.0f, 0.0f)

        testForward(input, 1, 2, expected)

    }

}