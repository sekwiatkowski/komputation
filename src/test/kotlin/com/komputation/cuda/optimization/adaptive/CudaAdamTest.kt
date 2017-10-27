package com.komputation.cuda.optimization.adaptive

import com.komputation.cuda.getFloatArray
import com.komputation.cuda.setFloatArray
import com.komputation.cuda.setUpCudaContext
import com.komputation.optimization.adaptive.adam
import jcuda.Pointer
import jcuda.runtime.JCuda.cudaFree
import org.junit.jupiter.api.Assertions.assertArrayEquals
import org.junit.jupiter.api.Test

class CudaAdamTest {

    @Test
    fun testOneDimension() {

        val cudaContext = setUpCudaContext()

        val numberRows = 1
        val numberColumns = 1
        val size = numberRows * numberColumns

        val adadelta = adam()
            .buildForCuda(cudaContext)
            .invoke(1, numberRows, numberColumns)

        adadelta.acquire(1)

        val deviceParameter = Pointer()
        val pointerToDeviceParameter = Pointer.to(deviceParameter)
        setFloatArray(floatArrayOf(2.0f), size, deviceParameter)

        val deviceGradient = Pointer()
        val pointerToDeviceGradient = Pointer.to(deviceGradient)

        // updatedFirstMomentEstimate = 0.9 * 0.0 + 0.1 * 0.1 = 0.01
        // correctedFirstMomentEstimate = 0.01 / (1.0 - pow(0.9, 1.0))
        //                              = 0.01 / (1.0 - 0.9)
        //                              = 0.1
        // updatedSecondMomentEstimate = 0.999 * 0.0 + 0.001 * 0.1 * 0.1 = 1.0e-5
        // correctedSecondMomentEstimate = 1.0e-5 / (1.0 - pow(0.999, 1.0))
        //                               = 1.0e-5 / (1.0 - 0.999)
        //                               = 0.01
        // adaptedLearningRate = 0.001 / (sqrt(0.01) + 1e-8f) = 0.009999999
        // update = -0.1 * 0.009999999 = -0.0009999999
        // parameter = 2.0 - 0.0009999999 = 1.9990000001
        setFloatArray(floatArrayOf(0.1f), size, deviceGradient)
        adadelta.denseUpdate(1, pointerToDeviceParameter, pointerToDeviceGradient)

        // updatedFirstMomentEstimate = 0.9 * 0.01 + 0.1 * 0.2 = 0.029
        // correctedFirstMomentEstimate = 0.029 / (1.0 - pow(0.9, 2.0)) = 0.15263157894
        // updatedSecondMomentEstimate = 0.999 * 1.0e-5 + 0.001 * 0.2 * 0.2 = 0.00004999
        // correctedSecondMomentEstimate = 0.00004999 / (1.0 - pow(0.999, 2.0)) = 0.02500750375
        // adaptedLearningRate = 0.001 / (sqrt(0.02500750375) + 1e-8f) = 0.00632360597
        // update = -0.15263157894 * 0.00632360597 = 0.00096518196
        // parameter = 1.9990000001 - 0.00096518196 = 1.99803481814
        setFloatArray(floatArrayOf(0.2f), size, deviceGradient)
        adadelta.denseUpdate(1, pointerToDeviceParameter, pointerToDeviceGradient)

        val hostParameter = getFloatArray(deviceParameter, size)

        cudaFree(deviceParameter)
        cudaFree(deviceGradient)

        adadelta.release()

        cudaContext.destroy()

        assertArrayEquals(floatArrayOf(1.99803481814f), hostParameter, 1e-6f)

    }

}