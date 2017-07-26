package shape.komputation.cuda.layers.forward

import jcuda.Pointer
import jcuda.jcublas.cublasHandle
import org.junit.jupiter.api.Test
import shape.komputation.layers.forward.dropout.dropoutLayer
import java.util.*
import jcuda.runtime.JCuda.cudaFree
import shape.komputation.matrix.floatColumnVector
import org.junit.jupiter.api.Assertions.assertArrayEquals
import shape.komputation.cuda.*

class CudaDropoutLayerTest {

    @Test
    fun testTraining() {

        val numberEntries = 10_000
        val keepProbability = 0.5f

        val random = Random()
        val input = FloatArray(numberEntries) { random.nextFloat() }

        val cpuLayer = dropoutLayer(numberEntries, Random(1), keepProbability).buildForCpu()
        val cpuResult = cpuLayer.forward(floatColumnVector(*input), true).entries

        val cudaResult = runForward(input, numberEntries, keepProbability, true)

        assertArrayEquals(cpuResult, cudaResult, 0.001f)

    }

    @Test
    fun testRuntime() {

        val input = floatArrayOf(1.0f, 2.0f)
        val keepProbability = 0.5f

        val expected = floatArrayOf(0.5f, 1.0f)
        val actual = runForward(input, input.size, keepProbability, false)

        assertArrayEquals(expected, actual, 0.001f)

    }


    private fun runForward(input: FloatArray, numberEntries: Int, keepProbability: Float, isTraining : Boolean): FloatArray {

        val cudaContext = setUpCudaContext()

        val cudaLayer = dropoutLayer(numberEntries, Random(1), keepProbability).buildForCuda(cudaContext, cublasHandle())

        cudaLayer.acquire()

        val deviceInput = Pointer()
        setFloatArray(input, numberEntries, deviceInput)

        val deviceResult = cudaLayer.forward(deviceInput, isTraining)
        val cudaResult = getFloatArray(deviceResult, numberEntries)

        cudaLayer.release()

        cudaFree(deviceInput)

        cudaContext.destroy()

        return cudaResult

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

        cudaLayer.acquire()

        val deviceInput = Pointer()
        allocateDeviceFloatMemory(deviceInput, numberEntries)

        val deviceChain = Pointer()
        setFloatArray(chain, numberEntries, deviceChain)

        cudaLayer.forward(deviceInput, true)
        val deviceResult = cudaLayer.backward(deviceChain)

        val cudaResult = getFloatArray(deviceResult, numberEntries)

        cudaLayer.release()

        cudaFree(deviceInput)

        cudaContext.destroy()

        return cudaResult

    }

}