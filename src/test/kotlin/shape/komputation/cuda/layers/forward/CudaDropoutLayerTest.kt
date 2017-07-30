package shape.komputation.cuda.layers.forward

import jcuda.Pointer
import jcuda.jcublas.cublasHandle
import jcuda.runtime.JCuda.cudaFree
import org.junit.jupiter.api.Assertions.assertArrayEquals
import org.junit.jupiter.api.Test
import shape.komputation.cuda.allocateDeviceFloatMemory
import shape.komputation.cuda.getFloatArray
import shape.komputation.cuda.setFloatArray
import shape.komputation.cuda.setUpCudaContext
import shape.komputation.layers.forward.dropout.dropoutLayer
import shape.komputation.matrix.floatColumnVector
import java.util.*

class CudaDropoutLayerTest {

    @Test
    fun testTraining() {

        val numberEntries = 10_000
        val keepProbability = 0.5f

        forward(numberEntries, keepProbability, true)

    }

    @Test
    fun testRuntime() {

        val numberEntries = 10_000
        val keepProbability = 0.5f

        forward(numberEntries, keepProbability, false)

    }


    private fun forward(numberEntries: Int, keepProbability: Float, isTraining : Boolean) {

        val random = Random()
        val input = FloatArray(numberEntries) { random.nextFloat() }

        val cpuLayer = dropoutLayer(numberEntries, Random(1), keepProbability).buildForCpu()
        val cpuResult = cpuLayer.forward(floatColumnVector(*input), isTraining).entries

        val cudaContext = setUpCudaContext()

        val cudaLayer = dropoutLayer(numberEntries, Random(1), keepProbability).buildForCuda(cudaContext, cublasHandle())

        cudaLayer.acquire(1)

        val deviceInput = Pointer()
        setFloatArray(input, numberEntries, deviceInput)

        val deviceResult = cudaLayer.forward(deviceInput, 1, isTraining)
        val cudaResult = getFloatArray(deviceResult, numberEntries)

        cudaLayer.release()

        cudaFree(deviceInput)

        cudaContext.destroy()

        assertArrayEquals(cpuResult, cudaResult, 0.001f)

    }

    @Test
    fun testBackward1() {

        val chain = floatArrayOf(1.0f, 2.0f)
        val expected = floatArrayOf(1.0f, 2.0f)

        val actual = runBackward(chain, chain.size, true)

        assertArrayEquals(expected, actual, 0.001f)

    }

    @Test
    fun testBackward2() {

        val chain = floatArrayOf(1.0f, 2.0f)
        val expected = floatArrayOf(0.0f, 0.0f)

        val actual = runBackward(chain, chain.size, false)

        assertArrayEquals(expected, actual, 0.001f)

    }


    private fun runBackward(chain : FloatArray, numberEntries: Int, keep : Boolean): FloatArray {

        val cudaContext = setUpCudaContext()

        val cudaLayer = dropoutLayer(numberEntries, Random(1), if(keep) 1.0f else 0.0f).buildForCuda(cudaContext, cublasHandle())

        cudaLayer.acquire(1)

        val deviceInput = Pointer()
        allocateDeviceFloatMemory(deviceInput, numberEntries)

        val deviceChain = Pointer()
        setFloatArray(chain, numberEntries, deviceChain)

        cudaLayer.forward(deviceInput, 1, true)
        val deviceResult = cudaLayer.backward(deviceChain, 1)

        val cudaResult = getFloatArray(deviceResult, numberEntries)

        cudaLayer.release()

        cudaFree(deviceInput)

        cudaContext.destroy()

        return cudaResult

    }

}