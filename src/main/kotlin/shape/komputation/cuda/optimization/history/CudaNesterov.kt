
package shape.komputation.cuda.optimization.history

import jcuda.Pointer
import jcuda.runtime.JCuda.cudaFree
import shape.komputation.cuda.allocateDeviceFloatMemory
import shape.komputation.cuda.kernels.Kernel
import shape.komputation.cuda.kernels.computeEntrywiseLaunchConfiguration
import shape.komputation.cuda.optimization.CudaUpdateRule
import shape.komputation.cuda.setIntArray
import shape.komputation.layers.Resourceful

class CudaNesterov(
    private val numberParameters : Int,
    private val parameterSize : Int,
    private val learningRate: Float,
    private val momentum : Float,
    private val createKernel: () -> Kernel,
    private val numberMultiprocessors : Int,
    private val numberResidentWarps : Int,
    private val warpSize : Int,
    private val maximumNumberThreads : Int) : CudaUpdateRule, Resourceful {

    private val pointerToParameterSize = Pointer.to(intArrayOf(this.parameterSize))
    private val pointerToLearningRate = Pointer.to(floatArrayOf(this.learningRate))
    private val pointerToMomentum = Pointer.to(floatArrayOf(this.momentum))

    private val deviceHistory = Pointer()
    private val pointerToHistory = Pointer.to(this.deviceHistory)

    private val deviceBackup = Pointer()
    private val pointerToBackup = Pointer.to(this.deviceBackup)

    private var kernel : Kernel? = null
    private var numberBlocks = -1
    private var numberThreads = -1
    private val numberIterations = intArrayOf(-1)
    private var pointerToNumberIterations = Pointer.to(this.numberIterations)

    private val deviceArrayOfZero = Pointer()
    private val pointToArrayOfZero = Pointer.to(this.deviceArrayOfZero)

    override fun acquire(maximumBatchSize : Int) {

        this.kernel = this.createKernel()

        val launchConfiguration = computeEntrywiseLaunchConfiguration(this.parameterSize, this.numberMultiprocessors, this.numberResidentWarps, this.warpSize, this.maximumNumberThreads)
        this.numberBlocks = launchConfiguration.numberBlocks
        this.numberThreads = launchConfiguration.numberThreadsPerBlock
        this.numberIterations[0] = launchConfiguration.numberIterations

        setIntArray(intArrayOf(0), 1, this.deviceArrayOfZero)

        val totalNumberEntries = this.numberParameters * this.parameterSize

        allocateDeviceFloatMemory(this.deviceHistory, totalNumberEntries)
        
        allocateDeviceFloatMemory(this.deviceBackup, totalNumberEntries)

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
            this.pointerToMomentum,
            this.pointerToHistory,
            this.pointerToBackup,
            pointerToParameterIndices,
            this.pointerToParameterSize,
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

        cudaFree(this.deviceHistory)
        cudaFree(this.deviceBackup)
        cudaFree(this.deviceArrayOfZero)

    }

}