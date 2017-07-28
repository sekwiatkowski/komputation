package shape.komputation.cuda.layers.forward.projection

import jcuda.Pointer
import jcuda.jcublas.JCublas2.cublasCreate
import jcuda.jcublas.JCublas2.cublasDestroy
import jcuda.jcublas.cublasHandle
import org.junit.jupiter.api.Assertions.assertArrayEquals
import org.junit.jupiter.api.Test
import shape.komputation.cuda.getFloatArray
import shape.komputation.cuda.setFloatArray
import shape.komputation.cuda.setUpCudaContext

class CublasBiasLayerTest {

    @Test
    fun testForward() {

        val expected = floatArrayOf(2.0f, 4.0f, 4.0f, 6.0f, 0f, 0f, 0f, 0f)

        val cudaContext = setUpCudaContext()
        val cublasHandle = cublasHandle()

        cublasCreate(cublasHandle)

        val numberInputRows = 2
        val numberInputColumns = 2
        val numberEntries = numberInputRows * numberInputColumns
        val biasLayer = CublasBiasLayer(null, cublasHandle, numberInputRows, cudaContext.maximumNumberThreadsPerBlock, numberInputColumns, { cudaContext.kernelFactory.bias() }, floatArrayOf(1.0f, 2.0f), null)

        val maximumBatchSize = 2
        biasLayer.acquire(maximumBatchSize)

        val deviceInput = Pointer()
        setFloatArray(floatArrayOf(1.0f, 2.0f, 3.0f, 4.0f), 4, deviceInput)
        val deviceResult = biasLayer.forward(deviceInput, 1, true)

        val actual = getFloatArray(deviceResult, maximumBatchSize * numberEntries)

        biasLayer.release()

        cublasDestroy(cublasHandle)
        cudaContext.destroy()

        assertArrayEquals(expected, actual, 0.001f)

    }

    @Test
    fun testBackward() {

        val expected = floatArrayOf(4f, 6f)

        val cudaContext = setUpCudaContext()
        val cublasHandle = cublasHandle()

        cublasCreate(cublasHandle)

        val numberInputRows = 2
        val numberInputColumns = 2
        val biasLayer = CublasBiasLayer(null, cublasHandle, numberInputRows, cudaContext.maximumNumberThreadsPerBlock, numberInputColumns, { cudaContext.kernelFactory.bias() }, floatArrayOf(1.0f, 2.0f), null)

        val maximumBatchSize = 2
        biasLayer.acquire(maximumBatchSize)

        val devieChain = Pointer()
        setFloatArray(floatArrayOf(1.0f, 2.0f, 3.0f, 4.0f, 0f, 0f, 0f, 0f), 4, devieChain)
        val deviceResult = biasLayer.backward(devieChain, 1)

        val actual = getFloatArray(deviceResult, numberInputColumns)

        biasLayer.release()

        cublasDestroy(cublasHandle)
        cudaContext.destroy()

        assertArrayEquals(expected, actual, 0.001f)

    }


}