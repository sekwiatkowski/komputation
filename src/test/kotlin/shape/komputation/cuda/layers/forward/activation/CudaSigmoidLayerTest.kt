package shape.komputation.cuda.layers.forward.activation

import jcuda.Pointer
import jcuda.jcublas.cublasHandle
import jcuda.runtime.JCuda.cudaFree
import org.junit.jupiter.api.Assertions.assertArrayEquals
import org.junit.jupiter.api.Test
import shape.komputation.cpu.functions.activation.sigmoid
import shape.komputation.cuda.getVector
import shape.komputation.cuda.setUpCudaContext
import shape.komputation.cuda.setVector
import shape.komputation.layers.forward.activation.sigmoidLayer

class CudaSigmoidLayerTest {

    @Test
    fun testForwardOneDimension() {

        val context = setUpCudaContext()

        val layer = sigmoidLayer(1).buildForCuda(context, cublasHandle())
        layer.acquire()

        val deviceInput = Pointer()
        setVector(doubleArrayOf(0.0), 1, deviceInput)

        val deviceResult = layer.forward(deviceInput)

        val actual = getVector(deviceResult, 1)

        val expected = doubleArrayOf(0.5)

        cudaFree(deviceInput)

        layer.release()

        assertArrayEquals(expected, actual, 0.001)

    }

    @Test
    fun testForwardTwoDimensions() {

        val context = setUpCudaContext()

        val layer = sigmoidLayer(2).buildForCuda(context, cublasHandle())
        layer.acquire()

        val deviceInput = Pointer()
        setVector(doubleArrayOf(0.0, 1.0), 2, deviceInput)

        val deviceResult = layer.forward(deviceInput)

        val actual = getVector(deviceResult, 2)

        val expected = doubleArrayOf(0.5, 0.731058579)

        cudaFree(deviceInput)

        layer.release()

        assertArrayEquals(expected, actual, 0.001)


    }

    @Test
    fun testBackwardOneDimension() {

        val context = setUpCudaContext()

        val layer = sigmoidLayer(1).buildForCuda(context, cublasHandle())
        layer.acquire()

        val input = Pointer()
        setVector(doubleArrayOf(0.0), 1, input)
        layer.forward(input)

        val chain = Pointer()
        setVector(doubleArrayOf(1.0), 1, chain)
        val deviceResult = layer.backward(chain)

        val actual = getVector(deviceResult, 1)

        cudaFree(input)
        cudaFree(chain)

        layer.release()

        val expected = doubleArrayOf(0.25)

        assertArrayEquals(expected, actual, 0.001)

    }

    @Test
    fun testBackwardTwoDimensions() {

        val context = setUpCudaContext()

        val layer = sigmoidLayer(2).buildForCuda(context, cublasHandle())
        layer.acquire()

        val deviceInput = Pointer()
        setVector(doubleArrayOf(0.0, 1.0), 2, deviceInput)
        layer.forward(deviceInput)

        val deviceChain = Pointer()
        setVector(doubleArrayOf(1.0, 2.0), 2, deviceChain)
        val deviceResult = layer.backward(deviceChain)

        val actual = getVector(deviceResult, 2)

        cudaFree(deviceInput)
        cudaFree(deviceChain)

        layer.release()

        val expected = doubleArrayOf(1 * 0.5 * (1 - 0.5), 2 * sigmoid(1.0) * (1.0 - sigmoid(1.0)))

        assertArrayEquals(expected, actual, 0.001)

    }


}