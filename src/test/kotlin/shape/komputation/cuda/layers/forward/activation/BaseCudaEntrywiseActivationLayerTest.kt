package shape.komputation.cuda.layers.forward.activation

import jcuda.Pointer
import jcuda.runtime.JCuda.cudaFree
import org.junit.jupiter.api.Assertions.assertArrayEquals
import shape.komputation.cuda.CudaContext
import shape.komputation.cuda.getFloatArray
import shape.komputation.cuda.setFloatArray
import shape.komputation.cuda.setUpCudaContext

abstract class BaseCudaEntrywiseActivationLayerTest {

    protected abstract fun createLayer(context: CudaContext, numberEntries: Int) : BaseCudaEntrywiseActivationLayer

    protected fun testForward(input: FloatArray, batchSize : Int, maximumBatchSize : Int, expected: FloatArray) {

        val numberEntries = input.size / maximumBatchSize

        val cudaContext = setUpCudaContext()

        val layer = createLayer(cudaContext, numberEntries)

        layer.acquire(maximumBatchSize)

        val deviceInput = Pointer()
        setFloatArray(input, numberEntries, deviceInput)

        val deviceResult = layer.forward(deviceInput, batchSize, true)
        val actual = getFloatArray(deviceResult, numberEntries * maximumBatchSize)

        cudaFree(deviceInput)

        layer.release()

        cudaContext.destroy()

        assertArrayEquals(expected, actual, 0.001f)

    }

    protected fun testBackward(input: FloatArray, chain: FloatArray, expected: FloatArray) {

        val numberEntries = input.size

        val context = setUpCudaContext()

        val layer = createLayer(context, numberEntries)
        layer.acquire(1)

        val deviceInput = Pointer()
        setFloatArray(input, numberEntries, deviceInput)
        layer.forward(deviceInput, 1, true)

        val deviceChain = Pointer()
        setFloatArray(chain, numberEntries, deviceChain)
        val deviceResult = layer.backward(deviceChain, 1)

        val actual = getFloatArray(deviceResult, numberEntries)

        cudaFree(deviceInput)
        cudaFree(deviceChain)

        layer.release()

        assertArrayEquals(expected, actual, 0.001f)

    }


}