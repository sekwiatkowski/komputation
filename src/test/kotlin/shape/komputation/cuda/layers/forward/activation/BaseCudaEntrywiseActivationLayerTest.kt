package shape.komputation.cuda.layers.forward.activation

import jcuda.Pointer
import jcuda.runtime.JCuda
import org.junit.jupiter.api.Assertions.assertArrayEquals
import shape.komputation.cuda.CudaContext
import shape.komputation.cuda.getFloatArray
import shape.komputation.cuda.setUpCudaContext
import shape.komputation.cuda.setFloatArray

abstract class BaseCudaEntrywiseActivationLayerTest {

    protected abstract fun createLayer(context: CudaContext, numberEntries: Int) : BaseCudaEntrywiseActivationLayer

    protected fun testForward(input: FloatArray, expected: FloatArray) {

        val numberEntries = input.size

        val cudaContext = setUpCudaContext()

        val layer = createLayer(cudaContext, numberEntries)

        layer.acquire()

        val deviceInput = Pointer()
        setFloatArray(input, numberEntries, deviceInput)

        val deviceResult = layer.forward(deviceInput, true)
        val actual = getFloatArray(deviceResult, numberEntries)

        JCuda.cudaFree(deviceInput)

        layer.release()

        cudaContext.destroy()

        assertArrayEquals(expected, actual, 0.001f)

    }

    protected fun testBackward(input: FloatArray, chain: FloatArray, expected: FloatArray) {

        val numberEntries = input.size

        val context = setUpCudaContext()

        val layer = createLayer(context, numberEntries)
        layer.acquire()

        val deviceInput = Pointer()
        setFloatArray(input, numberEntries, deviceInput)
        layer.forward(deviceInput, true)

        val deviceChain = Pointer()
        setFloatArray(chain, numberEntries, deviceChain)
        val deviceResult = layer.backward(deviceChain)

        val actual = getFloatArray(deviceResult, numberEntries)

        JCuda.cudaFree(deviceInput)
        JCuda.cudaFree(deviceChain)

        layer.release()

        assertArrayEquals(expected, actual, 0.001f)

    }


}