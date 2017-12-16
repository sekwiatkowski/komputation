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

fun main(args: Array<String>) {

    /* val context = setUpCudaContext()

    val cublasHandle = cublasHandle()
    cublasCreate(cublasHandle)

    val deviceInput = Pointer()
    setFloatArray(floatArrayOf(1f, 2f), 2, deviceInput)

    val cublasWeightingLayer = CublasWeightingLayer(null, cublasHandle, 2, 1, 2, floatArrayOf(1f, 2f, 3f, 4f))
    cublasWeightingLayer.acquire(1)
    val cudaRecurrentLayer = CudaRecurrentLayer(null, cublasWeightingLayer, { context.createKernel(RecurrentKernels.kernel()) }, floatArrayOf(1f, 2f), ActivationFunction.ReLU)
    cudaRecurrentLayer.acquire(1)

    val deviceResult = cudaRecurrentLayer.forward(1, deviceInput, false)

    val floatArray = getFloatArray(deviceResult, 2)

    cudaRecurrentLayer.release()
    cublasWeightingLayer.release()

    context.destroy()

    println("recurrent") */

    JCudaDriver.setExceptionsEnabled(true)

    val src = """
        extern "C"
        __global__ void child() {
            printf("child");
        }

        extern "C"
        __global__ void parent() {
            child<<< 1, 1>>>();
        }
    """

    val program = nvrtcProgram()
    val ptx = compileKernel(
        program,
        3 to 5,
        src,
        "dynamic_parallelism.cu",
        emptyArray(),
        emptyArray(),
        emptyArray()
    )

    val context = setUpCudaContext()

    val linkState = CUlinkState()
    val cubinPointer = link(linkState, ptx, "dynamic_parallelism.ptx", "C:\\Program Files\\NVIDIA GPU Computing Toolkit\\CUDA\\v8.0\\lib\\x64\\cudadevrt.lib")

    val kernel = CUfunction()
    loadKernel(kernel, cubinPointer, "parent")

    launchKernel(
        kernel,
        Pointer(),
        1,
        1,
        1,
        0
    )

    cudaDeviceSynchronize()

    nvrtcDestroyProgram(program)

    context.destroy()

}