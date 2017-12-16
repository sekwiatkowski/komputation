package com.komputation.cuda.layers.recurrent

import com.komputation.cuda.allocateDeviceFloatMemory
import com.komputation.cuda.getFloatArray
import com.komputation.cuda.kernels.Kernel
import com.komputation.cuda.layers.BaseCudaForwardLayer
import com.komputation.cuda.layers.forward.projection.CublasWeightingLayer
import com.komputation.cuda.setFloatArray
import com.komputation.cuda.setUpCudaContext
import com.komputation.layers.Resourceful
import com.komputation.layers.forward.activation.ActivationFunction
import jcuda.Pointer
import jcuda.driver.*
import jcuda.driver.CUjitInputType.CU_JIT_INPUT_LIBRARY
import jcuda.driver.CUjitInputType.CU_JIT_INPUT_PTX
import jcuda.driver.JCudaDriver.*
import jcuda.nvrtc.JNvrtc.*
import jcuda.nvrtc.nvrtcProgram
import jcuda.runtime.JCuda.cudaDeviceSynchronize
import jcuda.runtime.JCuda.cudaFree
import java.nio.ByteBuffer


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

    val prog = nvrtcProgram()

    nvrtcCreateProgram(prog, src, "dynamic_parallelism.cu", 0, emptyArray(), emptyArray())

    nvrtcCompileProgram(prog, 1, arrayOf("--gpu-architecture=compute_35"))

    val programLogArray = Array(1) { "" }
    nvrtcGetProgramLog(prog, programLogArray)

    val ptxArray = Array(1) { "" }
    nvrtcGetPTX(prog, ptxArray)

    nvrtcDestroyProgram(prog)

    val context = setUpCudaContext()

    val linkState = CUlinkState()
    cuLinkCreate(JITOptions(), linkState)

    cuLinkAddFile(linkState, CU_JIT_INPUT_LIBRARY, "C:\\Program Files\\NVIDIA GPU Computing Toolkit\\CUDA\\v8.0\\lib\\x64\\cudadevrt.lib", JITOptions())

    val ptx = ptxArray.single()
    val bytes = ptx.toByteArray()
    val byteBuffer = ByteBuffer.wrap(bytes)
    cuLinkAddData(linkState, CU_JIT_INPUT_PTX, Pointer.to(byteBuffer), bytes.size.toLong(),"dynamic_parallelism.ptx", JITOptions())

    val cubinPointer = Pointer()
    val linkSize = LongArray(1)
    cuLinkComplete(linkState, cubinPointer, linkSize)

    // https://stackoverflow.com/questions/32535828/jit-in-jcuda-loading-multiple-ptx-modules
    val module = CUmodule()
    cuModuleLoadDataEx(module, cubinPointer, 0, IntArray(0), Pointer.to(IntArray(0)))

    val function = CUfunction()
    cuModuleGetFunction(function, module, "parent")

    cuLaunchKernel(
        function,
        1, 1, 1,
        1, 1, 1,
        0,
        null,
        Pointer(),
        null
    )

    cudaDeviceSynchronize()

    context.destroy()

}