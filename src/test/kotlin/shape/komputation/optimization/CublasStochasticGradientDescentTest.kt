package shape.komputation.optimization

import jcuda.Pointer
import jcuda.Sizeof
import jcuda.jcublas.JCublas2.*
import jcuda.jcublas.cublasHandle
import jcuda.runtime.JCuda.cudaFree
import org.junit.jupiter.api.Assertions.assertArrayEquals
import org.junit.jupiter.api.Test
import shape.komputation.cuda.copyFromHostToDevice

class CublasStochasticGradientDescentTest {

    @Test
    fun testOneDimension() {

        val cublasHandle = cublasHandle()

        cublasCreate(cublasHandle)

        val numberRows = 1
        val numberColumns = 1
        val size = numberRows * numberColumns

        val cublasStochasticGradientDescent = CublasStochasticGradientDescent(cublasHandle, numberRows, numberColumns, 0.1)

        val deviceParameter = copyFromHostToDevice(doubleArrayOf(2.0), size)
        val deviceGradient = copyFromHostToDevice(doubleArrayOf(0.1), size)

        cublasStochasticGradientDescent.update(deviceParameter, 1.0, deviceGradient)

        val hostParameter = DoubleArray(1)

        cublasGetVector(size, Sizeof.DOUBLE, deviceParameter, 1, Pointer.to(hostParameter), 1)

        cudaFree(deviceParameter)
        cudaFree(deviceGradient)

        cublasDestroy(cublasHandle)

        assertArrayEquals(doubleArrayOf(1.99), hostParameter)

    }

}