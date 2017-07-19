package shape.komputation.cuda.optimization

import jcuda.Pointer
import jcuda.Sizeof
import jcuda.jcublas.JCublas2.cublasGetVector
import jcuda.runtime.JCuda.cudaFree
import org.junit.jupiter.api.Assertions.assertArrayEquals
import org.junit.jupiter.api.Test
import shape.komputation.cuda.setUpCudaContext
import shape.komputation.cuda.setVector

class CudaStochasticGradientDescentTest {

    @Test
    fun testOneDimension() {

        val cudaContext = setUpCudaContext()

        val numberRows = 1
        val numberColumns = 1
        val size = numberRows * numberColumns

        val stochasticGradientDescent = CudaStochasticGradientDescent(cudaContext.kernelFactory.stochasticGradientDescent(), cudaContext.maximumNumberThreadsPerBlock, numberRows * numberColumns, 0.1)

        stochasticGradientDescent.acquire()

        val deviceParameter = Pointer()
        setVector(doubleArrayOf(2.0), size, deviceParameter)
        val deviceGradient = Pointer()
        setVector(doubleArrayOf(0.1), size, deviceGradient)

        stochasticGradientDescent.update(Pointer.to(deviceParameter), 1.0, Pointer.to(deviceGradient))

        val hostParameter = DoubleArray(1)

        cublasGetVector(size, Sizeof.DOUBLE, deviceParameter, 1, Pointer.to(hostParameter), 1)

        cudaFree(deviceParameter)
        cudaFree(deviceGradient)

        stochasticGradientDescent.release()

        cudaContext.destroy()

        assertArrayEquals(doubleArrayOf(1.99), hostParameter)

    }

}