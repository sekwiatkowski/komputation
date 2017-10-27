package com.komputation.cuda.optimization.adaptive

import com.komputation.cuda.getFloatArray
import com.komputation.cuda.setFloatArray
import com.komputation.cuda.setUpCudaContext
import com.komputation.optimization.adaptive.adadelta
import jcuda.Pointer
import jcuda.runtime.JCuda.cudaFree
import org.junit.jupiter.api.Assertions
import org.junit.jupiter.api.Test

class CudaAdadeltaTest {

    @Test
    fun testOneDimension() {

        val cudaContext = setUpCudaContext()

        val numberRows = 1
        val numberColumns = 1
        val size = numberRows * numberColumns

        val adadelta = adadelta()
            .buildForCuda(cudaContext)
            .invoke(1, numberRows, numberColumns)

        adadelta.acquire(1)

        val deviceParameter = Pointer()
        val pointerToDeviceParameter = Pointer.to(deviceParameter)
        setFloatArray(floatArrayOf(2.0f), size, deviceParameter)

        val deviceGradient = Pointer()
        val pointerToDeviceGradient = Pointer.to(deviceGradient)

        // newGradientAccumulation = 0.95 * 0.0 + 0.05 * (0.1 * 0.1) = 0.0005
        // rootMeanSquaredOfDerivatives = sqrt(0.0005 + 1e-6) = 0.0223830293
        // rootMeanSquaredOfPastUpdates = sqrt(0.0 + 1e-6) = 0.001
        // learningRate = rootMeanSquaredOfPastUpdates / rootMeanSquaredOfDerivatives = 0.001 / 0.0223830293 = 0.0446767051
        // update = -learningRate * derivative = -0.0446767051 * 0.1 = -0.00446767051
        // newUpdateAccumulation = decay * pastUpdateAccumulation + oneMinusDecay * (update * update) = 0.95 * 0.0 + 0.05 * -0.00446767051 * -0.00446767051 = 9.98003989e-7
        // parameter = 2.0 - 0.00446767051 = 1.99553233 ~ 1.9955323
        setFloatArray(floatArrayOf(0.1f), size, deviceGradient)
        adadelta.denseUpdate(1, pointerToDeviceParameter, pointerToDeviceGradient)

        // newGradientAccumulation = 0.95 * 0.0005 + 0.05 * (0.2 * 0.2) = 0.002475
        // rootMeanSquaredOfDerivatives = sqrt(0.002475 + 1e-6) = 0.0497594212
        // rootMeanSquaredOfPastUpdates = sqrt(9.98003989e-7 + 1e-6) = 0.00141350769
        // learningRate = rootMeanSquaredOfPastUpdates / rootMeanSquaredOfDerivatives = 0.00141350769 / 0.0497594212 = 0.0284068354
        // update = -learningRate * derivative = -0.0284068354 * 0.2 = -0.00568136708
        // newUpdateAccumulation = decay * pastUpdateAccumulation + oneMinusDecay * (update * update) = 0.95 * 9.98003989e-7 + 0.05 * 0.00568136708 * 0.00568136708 = 2.56200038e-6
        // parameter = 1.9955323 - 0.00568136708 = 1.98985093
        setFloatArray(floatArrayOf(0.2f), size, deviceGradient)
        adadelta.denseUpdate(1, pointerToDeviceParameter, pointerToDeviceGradient)

        val hostParameter = getFloatArray(deviceParameter, size)

        cudaFree(deviceParameter)
        cudaFree(deviceGradient)

        adadelta.release()

        cudaContext.destroy()

        Assertions.assertArrayEquals(floatArrayOf(1.9898509f), hostParameter)

    }

}