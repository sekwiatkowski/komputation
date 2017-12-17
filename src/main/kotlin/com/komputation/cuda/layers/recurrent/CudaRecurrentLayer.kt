package com.komputation.cuda.layers.recurrent

import com.komputation.cuda.allocateDeviceFloatMemory
import com.komputation.cuda.getFloatArray
import com.komputation.cuda.kernels.*
import com.komputation.cuda.layers.BaseCudaForwardLayer
import com.komputation.cuda.layers.forward.projection.CublasWeightingLayer
import com.komputation.cuda.setFloatArray
import com.komputation.cuda.setUpCudaContext
import com.komputation.layers.Resourceful
import com.komputation.layers.forward.activation.ActivationFunction
import jcuda.Pointer
import jcuda.driver.CUfunction
import jcuda.driver.CUlinkState
import jcuda.driver.JCudaDriver
import jcuda.nvrtc.JNvrtc.nvrtcDestroyProgram
import jcuda.nvrtc.nvrtcProgram
import jcuda.runtime.JCuda.cudaDeviceSynchronize
import jcuda.runtime.JCuda.cudaFree


class CudaRecurrentLayer(
    private val name : String?,
    private val inputWeighting : CublasWeightingLayer,
    private val createKernel: () -> Kernel,
    private val initialBias : FloatArray,
    private val activation : ActivationFunction) : BaseCudaForwardLayer(name), Resourceful {

    override val deviceForwardResult: Pointer
        get() = this.deviceResult
    override val numberOutputRows: Int
        get() = this.numberInputRows
    override val maximumOutputColumns: Int
        get() = this.maximumInputColumns
    override val deviceBackwardResult: Pointer
        get() = this.inputWeighting.deviceBackwardResult
    override val numberInputRows: Int
        get() = this.inputWeighting.numberInputRows
    override val maximumInputColumns: Int
        get() = this.inputWeighting.maximumInputColumns

    private var kernel : Kernel? = null
    private val deviceBias = Pointer()
    private val deviceResult = Pointer()

    private val hiddenDimension = this.inputWeighting.numberOutputRows

    override fun acquire(maximumBatchSize: Int) {
        this.kernel = this.createKernel()

        setFloatArray(this.initialBias, this.hiddenDimension, this.deviceBias)
        allocateDeviceFloatMemory(this.deviceResult, this.hiddenDimension)
    }

    override fun release() {
        this.kernel!!.destroy()

        cudaFree(this.deviceBias)
        cudaFree(this.deviceResult)
    }

    private val pointerToBias = Pointer.to(this.deviceBias)
    private val pointerToResult = Pointer.to(this.deviceResult)

    override fun forward(batchSize: Int, deviceInput: Pointer, isTraining: Boolean): Pointer {
        TODO("not implemented")

        val deviceWeightedInput = this.inputWeighting.forward(batchSize, deviceInput, isTraining)

        val floatArray = getFloatArray(deviceWeightedInput, 2)

        this.kernel!!.launch(
            Pointer.to(
                Pointer.to(deviceWeightedInput),
                this.pointerToBias,
                this.pointerToResult
            ),
            1,
            1,
            this.inputWeighting.numberInputRows,
            0
        )

        return deviceResult

    }

    override fun backward(batchSize: Int, chain: Pointer): Pointer {
        TODO("not implemented")
    }

}