package com.komputation.cuda.optimization.adaptive

import com.komputation.cuda.getFloatArray
import com.komputation.cuda.setFloatArray
import com.komputation.cuda.setUpCudaContext
import com.komputation.optimization.adaptive.adagrad
import jcuda.Pointer
import jcuda.runtime.JCuda.cudaFree
import org.junit.jupiter.api.Assertions.assertArrayEquals
import org.junit.jupiter.api.Test

class CudaAdagradTest {

    @Test
    fun testOneDimension() {

        val cudaContext = setUpCudaContext()

        val numberRows = 1
        val numberColumns = 1
        val size = numberRows * numberColumns

        val adagrad = adagrad(0.1f)
            .buildForCuda(cudaContext)
            .invoke(1, numberRows, numberColumns)

        adagrad.acquire(1)

        val deviceParameter = Pointer()
        val pointerToDeviceParameter = Pointer.to(deviceParameter)
        setFloatArray(floatArrayOf(2.0f), size, deviceParameter)

        val deviceGradient = Pointer()
        val pointerToDeviceGradient = Pointer.to(deviceGradient)

        // updatedHistory = 0.1^2 = 0.01
        // adaptedLearningRate = learning rate / (sqrt(updatedHistory) + epsilon)
        //                     = 0.1 / (sqrt(0.01) + 1e-6f)
        //                     = 0.99999
        // update = adaptedLearningRate * gradient = 0.99999 * 0.1 = 0.099999
        // parameter = parameter - update = 2.0 - 0.099999 = 1.900001
        setFloatArray(floatArrayOf(0.1f), size, deviceGradient)
        adagrad.denseUpdate(1, pointerToDeviceParameter, pointerToDeviceGradient)

        setFloatArray(floatArrayOf(0.2f), size, deviceGradient)
        // updatedHistory = 0.01 + 0.2^2 = 0.05
        // adaptedLearningRate = learningRate / (sqrt(updatedHistory) + epsilon)
        //                     = 0.1 / (sqrt(0.05) + 1e-6f)
        //                     ~ 0.447211596
        // update = adaptedLearningRate * gradient = 0.447211596 * 0.2 = 0.0894423192
        // parameter = parameter - update = 1.900001 - 0.0894423192 = 1.81055868
        adagrad.denseUpdate(1, pointerToDeviceParameter, pointerToDeviceGradient)

        val hostParameter = getFloatArray(deviceParameter, size)

        cudaFree(deviceParameter)
        cudaFree(deviceGradient)

        adagrad.release()

        cudaContext.destroy()

        assertArrayEquals(floatArrayOf(1.8105587f), hostParameter, 1e-6f)

    }

}