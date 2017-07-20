package shape.komputation.cuda.layers.forward.activation

import jcuda.jcublas.cublasHandle
import org.junit.jupiter.api.Test
import shape.komputation.cuda.CudaContext
import shape.komputation.layers.forward.activation.exponentiationLayer

class CudaExponentiationLayerTest : BaseCudaEntrywiseActivationLayerTest() {

    override fun createLayer(context: CudaContext, numberEntries: Int) =

        exponentiationLayer(numberEntries).buildForCuda(context, cublasHandle())

    @Test
    fun testForwardOneDimension() {

        val input = doubleArrayOf(1.0)
        val expected = doubleArrayOf(Math.exp(1.0))

        testForward(input, expected)

    }

    @Test
    fun testForwardTwoDimensions() {

        val input = doubleArrayOf(0.0, 1.0)
        val expected = doubleArrayOf(1.0, Math.exp(1.0))

        testForward(input, expected)

    }


    @Test
    fun testBackwardOneDimension() {

        val input = doubleArrayOf(1.0)
        val chain = doubleArrayOf(2.0)
        val expected = doubleArrayOf(2.0 * Math.exp(1.0))

        testBackward(input, chain, expected)

    }

    @Test
    fun testBackwardTwoDimensions() {

        val input = doubleArrayOf(0.0, 1.0)
        val chain = doubleArrayOf(2.0, 3.0)
        val expected = doubleArrayOf(2.0, 3.0 * Math.exp(1.0))

        testBackward(input, chain, expected)

    }


}