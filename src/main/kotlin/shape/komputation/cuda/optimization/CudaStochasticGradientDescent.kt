package shape.komputation.cuda.optimization

import jcuda.Pointer
import jcuda.runtime.JCuda.cudaFree
import shape.komputation.cuda.kernels.Kernel
import shape.komputation.cuda.kernels.launch.computeEntrywiseLaunchConfiguration
import shape.komputation.cuda.setIntArray
import shape.komputation.layers.Resourceful

class CudaStochasticGradientDescent internal constructor(
    private val size : Int,
    private val learningRate: Float,
    private val createKernel: () -> Kernel,
    private val numberMultiprocessors : Int,
    private val numberResidentWarps : Int,
    private val warpSize : Int,
    private val maximumNumberThreads : Int) : CudaUpdateRule, Resourceful {

    private val pointerToSize = Pointer.to(intArrayOf(this.size))
    private val pointerToLearningRate = Pointer.to(floatArrayOf(this.learningRate))

    private var kernel : Kernel? = null
    private var numberBlocks = -1
    private var numberThreads = -1
    private val numberIterations = intArrayOf(-1)
    private var pointerToNumberIterations = Pointer.to(this.numberIterations)

    private val deviceArrayOfZero = Pointer()
    private val pointToArrayOfZero = Pointer.to(this.deviceArrayOfZero)

    override fun acquire(maximumBatchSize : Int) {

        this.kernel = this.createKernel()

        val launchConfiguration = computeEntrywiseLaunchConfiguration(this.size, this.numberMultiprocessors, this.numberResidentWarps, this.warpSize, this.maximumNumberThreads)
        this.numberBlocks = launchConfiguration.numberBlocks
        this.numberThreads = launchConfiguration.numberThreadsPerBlock
        this.numberIterations[0] = launchConfiguration.numberIterations

        setIntArray(intArrayOf(0), 1, this.deviceArrayOfZero)

    }

    private val scalingFactorArray = floatArrayOf(Float.NaN)
    private val pointerToScalingFactor = Pointer.to(this.scalingFactorArray)

    override fun denseUpdate(pointerToParameters: Pointer, scalingFactor : Float, pointerToGradient: Pointer) {

        this.launchKernel(1, this.pointToArrayOfZero, pointerToParameters, scalingFactor, pointerToGradient)

    }

    override fun sparseUpdate(numberParameters : Int, pointerToParameterIndices: Pointer, pointerToParameters: Pointer, scalingFactor : Float, pointerToGradient: Pointer) {

        this.launchKernel(numberParameters, pointerToParameterIndices, pointerToParameters, scalingFactor, pointerToGradient)

    }

    private fun launchKernel(numberParameters: Int, pointerToParameterIndices : Pointer, pointerToDeviceParameter: Pointer, scalingFactor: Float, pointerToDeviceGradient: Pointer) {

        this.scalingFactorArray[0] = scalingFactor

        val parameters = Pointer.to(
            this.pointerToNumberIterations,
            this.pointerToLearningRate,
            pointerToParameterIndices,
            this.pointerToSize,
            pointerToDeviceParameter,
            this.pointerToScalingFactor,
            pointerToDeviceGradient
        )

        this.kernel!!.launch(
            parameters,
            numberParameters,
            this.numberBlocks,
            this.numberThreads,
            0
        )

    }

    override fun release() {

        this.kernel!!.destroy()

        cudaFree(this.deviceArrayOfZero)

    }

}