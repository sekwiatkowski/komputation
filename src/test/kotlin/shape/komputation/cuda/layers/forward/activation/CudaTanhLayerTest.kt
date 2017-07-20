package shape.komputation.cuda.layers.forward.activation

import jcuda.jcublas.cublasHandle
import org.junit.jupiter.api.Test
import shape.komputation.cuda.CudaContext
import shape.komputation.layers.forward.activation.tanhLayer

class CudaTanhLayerTest : BaseCudaEntrywiseActivationLayerTest() {

    override fun createLayer(context: CudaContext, numberEntries: Int) =

        tanhLayer(numberEntries).buildForCuda(context, cublasHandle())

    @Test
    fun testForwardOneDimension1() {

        val input = doubleArrayOf(0.0)
        val expected = doubleArrayOf(0.0)

        testForward(input, expected)

    }

    @Test
    fun testForwardOneDimension2() {

        val input = doubleArrayOf(1.0)
        val expected = doubleArrayOf(0.761594156)

        testForward(input, expected)

    }


    @Test
    fun testForwardOneDimension3() {

        val input = doubleArrayOf(-1.0)
        val expected = doubleArrayOf(-0.761594156)

        testForward(input, expected)

    }

    @Test
    fun testForwardTwoDimensions() {

        val input = doubleArrayOf(-2.0, 3.0)
        val expected = doubleArrayOf(-0.96402758, 0.995054754)

        testForward(input, expected)

    }

    @Test
    fun testBackwardOneDimension() {

        val input = doubleArrayOf(1.0)
        val chain = doubleArrayOf(1.0)
        val expected = doubleArrayOf(1.0 * (1.0 - Math.pow(Math.tanh(1.0), 2.0)))

        testBackward(input, chain, expected)

    }

}