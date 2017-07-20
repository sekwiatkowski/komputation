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

        val input = doubleArrayOf(0.0)
        val expected = doubleArrayOf(0.5)

        testForward(input, expected)

    }

    @Test
    fun testForwardTwoDimensions() {

        val input = doubleArrayOf(0.0, 1.0)
        val expected = doubleArrayOf(0.5, 0.731058579)

        testForward(input, expected)

    }


    @Test
    fun testBackwardOneDimension() {

        val input = doubleArrayOf(0.0)
        val chain = doubleArrayOf(1.0)
        val expected = doubleArrayOf(0.25)

        testBackward(input, chain, expected)

    }

    @Test
    fun testBackwardTwoDimensions() {

        val input = doubleArrayOf(0.0, 1.0)
        val chain = doubleArrayOf(1.0, 2.0)
        val expected = doubleArrayOf(1 * 0.5 * (1 - 0.5), 2 * sigmoid(1.0) * (1.0 - sigmoid(1.0)))

        testBackward(input, chain, expected)

    }

}