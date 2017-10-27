package com.komputation.cuda.optimization.adaptive

import com.komputation.cuda.getFloatArray
import com.komputation.cuda.setFloatArray
import com.komputation.cuda.setUpCudaContext
import com.komputation.optimization.adaptive.rmsprop
import jcuda.Pointer
import jcuda.runtime.JCuda.cudaFree
import org.junit.jupiter.api.Assertions.assertArrayEquals
import org.junit.jupiter.api.Test

class CudaRMSPropTest {

    @Test
    fun testOneDimension() {

        val cudaContext = setUpCudaContext()

        val numberRows = 1
        val numberColumns = 1
        val size = numberRows * numberColumns

        val adadelta = rmsprop(0.1f)
            .buildForCuda(cudaContext)
            .invoke(1, numberRows, numberColumns)

        adadelta.acquire(1)

        val deviceParameter = Pointer()
        val pointerToDeviceParameter = Pointer.to(deviceParameter)
        setFloatArray(floatArrayOf(2.0f), size, deviceParameter)

        val deviceGradient = Pointer()
        val pointerToDeviceGradient = Pointer.to(deviceGradient)

        // updatedAccumulation = 0.9 * 0.0 + 0.1 * (0.1 * 0.1) = 0.001
        // adaptiveLearningRate = 0.1 / sqrt(0.001 + 1e-6f) = 3.16069771
        // update = -3.16069771 * 0.1 = -0.316069771
        // parameter = 2.0 - 0.316069771 ~ 1.6839303
        setFloatArray(floatArrayOf(0.1f), size, deviceGradient)
        adadelta.denseUpdate(1, pointerToDeviceParameter, pointerToDeviceGradient)

        // updatedAccumulation = 0.9 * 0.001 + 0.1 * (0.2 * 0.2) = 0.0049
        // adaptiveLearningRate = 0.1 / sqrt(0.0049 + 1e-6f) = 1.42842568
        // update = -1.42842568 * 0.2 = -0.285685136
        // parameter = 1.6839303 - 0.285685136 = 1.39824516
        setFloatArray(floatArrayOf(0.2f), size, deviceGradient)
        adadelta.denseUpdate(1, pointerToDeviceParameter, pointerToDeviceGradient)

        val hostParameter = getFloatArray(deviceParameter, size)

        cudaFree(deviceParameter)
        cudaFree(deviceGradient)

        adadelta.release()

        cudaContext.destroy()

        assertArrayEquals(floatArrayOf(1.3982452f), hostParameter, 1e-6f)

    }

}