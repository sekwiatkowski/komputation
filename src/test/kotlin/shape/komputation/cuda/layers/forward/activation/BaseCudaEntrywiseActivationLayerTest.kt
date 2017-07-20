package shape.komputation.cuda.layers.forward.activation

import jcuda.Pointer
import jcuda.runtime.JCuda
import org.junit.jupiter.api.Assertions
import shape.komputation.cuda.CudaContext
import shape.komputation.cuda.getVector
import shape.komputation.cuda.setUpCudaContext
import shape.komputation.cuda.setVector

abstract class BaseCudaEntrywiseActivationLayerTest {

    protected abstract fun createLayer(context: CudaContext, numberEntries: Int) : CudaActivationLayer

    protected fun testForward(input: DoubleArray, expected: DoubleArray) {

        val numberEntries = input.size

        val cudaContext = setUpCudaContext()

        val layer = createLayer(cudaContext, numberEntries)

        layer.acquire()

        val deviceInput = Pointer()
        setVector(input, numberEntries, deviceInput)

        val deviceResult = layer.forward(deviceInput)
        val actual = getVector(deviceResult, numberEntries)

        JCuda.cudaFree(deviceInput)

        layer.release()

        cudaContext.destroy()

        Assertions.assertArrayEquals(expected, actual, 0.001)

    }

    protected fun testBackward(input: DoubleArray, chain: DoubleArray, expected: DoubleArray) {

        val numberEntries = input.size

        val context = setUpCudaContext()

        val layer = createLayer(context, numberEntries)
        layer.acquire()

        val deviceInput = Pointer()
        setVector(input, numberEntries, deviceInput)
        layer.forward(deviceInput)

        val deviceChain = Pointer()
        setVector(chain, numberEntries, deviceChain)
        val deviceResult = layer.backward(deviceChain)

        val actual = getVector(deviceResult, numberEntries)

        JCuda.cudaFree(deviceInput)
        JCuda.cudaFree(deviceChain)

        layer.release()

        Assertions.assertArrayEquals(expected, actual, 0.001)

    }


}