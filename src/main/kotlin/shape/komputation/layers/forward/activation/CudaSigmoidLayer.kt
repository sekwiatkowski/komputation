package shape.komputation.layers.forward.activation

import jcuda.Pointer
import jcuda.driver.CUfunction
import jcuda.driver.JCudaDriver.cuCtxSynchronize
import jcuda.runtime.JCuda.cudaFree
import shape.komputation.cuda.*
import shape.komputation.layers.Resourceful
import shape.komputation.matrix.DoubleMatrix
import java.io.File

class CudaSigmoidLayer internal constructor(
    name : String? = null,
    private val computeCapabilities: Pair<Int, Int>,
    maximumThreadsPerBlock: Int,
    private val inputDimension : Int) : ActivationLayer(name), Resourceful {

    private val resultDimension = inputDimension

    private val deviceInput = Pointer()

    private val numberThreads = Math.min(this.inputDimension, maximumThreadsPerBlock)
    private val numberBlocks = Math.ceil(inputDimension.toDouble() / numberThreads.toDouble()).toInt()

    private val deviceForwardResult = Pointer()

    private var forwardPtxFile: File? = null
    private val forwardFunction = CUfunction()
    private val forwardParameters = Pointer.to(
        Pointer.to(intArrayOf(this.inputDimension)),
        Pointer.to(this.deviceInput),
        Pointer.to(this.deviceForwardResult)
    )

    private val deviceChain = Pointer()
    private val deviceBackwardResult = Pointer()

    private var backwardPtxFile: File? = null
    private val backwardFunction = CUfunction()
    private val backwardParameters = Pointer.to(
        Pointer.to(intArrayOf(this.inputDimension)),
        Pointer.to(this.deviceForwardResult),
        Pointer.to(this.deviceChain),
        Pointer.to(this.deviceBackwardResult)
    )

    override fun acquire() {

        this.forwardPtxFile = acquireKernel(
            File(this.javaClass.getResource("/cuda/Sigmoid.cu").toURI()),
            "sigmoidKernel",
            this.forwardFunction)

        this.backwardPtxFile = acquireKernel(
            File(this.javaClass.getResource("/cuda/BackwardSigmoid.cu").toURI()),
            "backwardSigmoidKernel",
            this.backwardFunction)

        allocateDeviceMemory(this.deviceInput, this.inputDimension)
        allocateDeviceMemory(this.deviceChain, this.inputDimension)
        allocateDeviceMemory(this.deviceForwardResult, this.resultDimension)
        allocateDeviceMemory(this.deviceBackwardResult, this.inputDimension)

    }

    private fun acquireKernel(cuFile : File, kernelName: String, kernel: CUfunction): File {

        val ptxFile = File.createTempFile(kernelName, ".ptx")
        ptxFile.deleteOnExit()

        val ptxPath = ptxFile.path

        compileKernel(cuFile.path, ptxPath, this.computeCapabilities)

        loadKernel(ptxPath, kernel, kernelName)

        return ptxFile

    }

    override fun forward(input : DoubleMatrix, isTraining : Boolean): DoubleMatrix {

        setVector(this.deviceInput, input.entries, this.inputDimension)

        launchKernel(this.forwardFunction, this.forwardParameters, this.numberBlocks, this.numberThreads)

        cuCtxSynchronize()

        val result = getVector(this.deviceForwardResult, this.resultDimension)

        return DoubleMatrix(this.resultDimension, 1, result)

    }

    override fun backward(chain : DoubleMatrix) : DoubleMatrix {

        setVector(this.deviceChain, chain.entries, this.inputDimension)

        launchKernel(this.backwardFunction, this.backwardParameters, this.numberBlocks, this.numberThreads)

        cuCtxSynchronize()

        val result = getVector(this.deviceBackwardResult, this.resultDimension)

        return DoubleMatrix(this.resultDimension, 1, result)

    }

    override fun release() {

        this.forwardPtxFile!!.delete()
        this.backwardPtxFile!!.delete()

        cudaFree(this.deviceInput)
        cudaFree(this.deviceForwardResult)
        cudaFree(this.deviceChain)
        cudaFree(this.deviceBackwardResult)

    }

}
fun cudaSigmoidLayer(
    environment: CudaEnvironment,
    inputDimension : Int) =

    cudaSigmoidLayer(null, environment, inputDimension)


fun cudaSigmoidLayer(
    name : String? = null,
    environment: CudaEnvironment,
    inputDimension : Int) =

    CudaSigmoidLayer(name, environment.computeCapabilities, environment.numberThreadsPerBlock, inputDimension)