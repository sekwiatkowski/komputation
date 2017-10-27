package com.komputation.cuda.optimization.historical

import com.komputation.cuda.getFloatArray
import com.komputation.cuda.setFloatArray
import com.komputation.cuda.setUpCudaContext
import com.komputation.optimization.historical.momentum
import jcuda.Pointer
import jcuda.runtime.JCuda.cudaFree
import org.junit.jupiter.api.Assertions.assertArrayEquals
import org.junit.jupiter.api.Test

class CudaMomentumTest {

    @Test
    fun testOneDimension() {

        val cudaContext = setUpCudaContext()

        val numberRows = 1
        val numberColumns = 1
        val size = numberRows * numberColumns

        val momentum = momentum(0.1f, 0.9f).buildForCuda(cudaContext).invoke(1, numberRows, numberColumns)

        momentum.acquire(1)

        val deviceParameter = Pointer()
        val pointerToDeviceParameter = Pointer.to(deviceParameter)
        setFloatArray(floatArrayOf(2.0f), size, deviceParameter)

        val deviceGradient = Pointer()
        val pointerToDeviceGradient = Pointer.to(deviceGradient)

        // history = scaling factor * learning rate * gradient = -1.0 * 0.1 * 0.1 = -0.01
        // parameter = 2.0 + history = 1.99
        setFloatArray(floatArrayOf(0.1f), size, deviceGradient)
        momentum.denseUpdate(1, pointerToDeviceParameter, pointerToDeviceGradient)

        // history = momentum * history - scaling factor * learning rate * gradient = 0.9 * (-0.01) - 1.0 * 0.1 * 0.2 = -0.009 - 0.02 = -0.029
        // parameter = 1.99 + history = 1.961
        setFloatArray(floatArrayOf(0.2f), size, deviceGradient)
        momentum.denseUpdate(1, pointerToDeviceParameter, pointerToDeviceGradient)

        val hostParameter = getFloatArray(deviceParameter, size)

        cudaFree(deviceParameter)
        cudaFree(deviceGradient)

        momentum.release()

        cudaContext.destroy()

        assertArrayEquals(floatArrayOf(1.961f), hostParameter,1e-6f)

    }

}