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

        val input = doubleArrayOf(1.0)
        val expected = doubleArrayOf(1.0)

        testForward(input, expected)

    }

    @Test
    fun testForwardOneDimension2() {

        val input = doubleArrayOf(-1.0)
        val expected = doubleArrayOf(0.0)

        testForward(input, expected)

    }

    @Test
    fun testForwardOneDimension3() {

        val input = doubleArrayOf(0.0)
        val expected = doubleArrayOf(0.0)

        testForward(input, expected)

    }


    @Test
    fun testForwardTwoDimensions() {

        val input = doubleArrayOf(-2.0, 2.0)
        val expected = doubleArrayOf(0.0, 2.0)

        testForward(input, expected)


    }

    @Test
    fun testBackwardOneDimension1() {

        val input = doubleArrayOf(1.0)
        val chain = doubleArrayOf(2.0)
        val expected = doubleArrayOf(2.0)

        testBackward(input, chain, expected)

    }

    @Test
    fun testBackwardOneDimension2() {

        val input = doubleArrayOf(-1.0)
        val chain = doubleArrayOf(2.0)
        val expected = doubleArrayOf(0.0)

        testBackward(input, chain, expected)

    }

    @Test
    fun testBackwardOneDimension3() {

        val input = doubleArrayOf(0.0)
        val chain = doubleArrayOf(2.0)
        val expected = doubleArrayOf(0.0)

        testBackward(input, chain, expected)

    }


    @Test
    fun testBackwardTwoDimensions() {

        val input = doubleArrayOf(-1.0, 1.0)
        val chain = doubleArrayOf(1.0, 2.0)
        val expected = doubleArrayOf(0.0, 2.0)

        testBackward(input, chain, expected)

    }


}