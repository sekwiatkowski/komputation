package com.komputation.cuda.optimization.adaptive

import com.komputation.cuda.allocateDeviceFloatMemory
import com.komputation.cuda.kernels.Kernel
import com.komputation.cuda.kernels.launch.computeEntrywiseLaunchConfiguration
import com.komputation.cuda.optimization.BaseCudaUpdateRule
import com.komputation.instructions.Resourceful
import jcuda.Pointer
import jcuda.runtime.JCuda.cudaFree

class CudaAdam internal constructor(
    private val numberParameters : Int,
    private val parameterSize : Int,
    private val learningRate: Float,
    private val firstMomentDecay: Float,
    private val secondMomentDecay: Float,
    private val epsilon : Float,
    private val createKernel: () -> Kernel,
    private val numberMultiprocessors : Int,
    private val numberResidentWarps : Int,
    private val warpSize : Int,
    private val maximumNumberThreads : Int) : BaseCudaUpdateRule(), Resourceful {

    private val pointerToParameterSize = Pointer.to(intArrayOf(this.parameterSize))

    private val pointerToLearningRate = Pointer.to(floatArrayOf(this.learningRate))

    private val pointerToFirstMomentDecay = Pointer.to(floatArrayOf(this.firstMomentDecay))
    private val pointerToOneMinusFirstMomentDecay = Pointer.to(floatArrayOf(1.0f - this.firstMomentDecay))

    private val pointerToSecondMomentDecay = Pointer.to(floatArrayOf(this.secondMomentDecay))
    private val pointerToOneMinusSecondMomentDecay = Pointer.to(floatArrayOf(1.0f - this.secondMomentDecay))

    private val pointerToEpsilon = Pointer.to(floatArrayOf(this.epsilon))

    private val step = floatArrayOf(0.0f)
    private val pointerToStep = Pointer.to(this.step)

    private val deviceFirstMomentEstimate = Pointer()
    private val pointerToFirstMomentEstimate = Pointer.to(this.deviceFirstMomentEstimate)

    private val deviceSecondMomentEstimate = Pointer()
    private val pointerToSecondMomentEstimate = Pointer.to(this.deviceSecondMomentEstimate)

    private var kernel : Kernel? = null
    private var numberBlocks = -1
    private var numberThreads = -1
    private val numberIterations = intArrayOf(-1)
    private var pointerToNumberIterations = Pointer.to(this.numberIterations)

    override fun acquire(maximumBatchSize : Int) {
        super.acquire(maximumBatchSize)

        this.kernel = this.createKernel()

        val launchConfiguration = computeEntrywiseLaunchConfiguration(this.parameterSize, this.numberMultiprocessors, this.numberResidentWarps, this.warpSize, this.maximumNumberThreads)
        this.numberBlocks = launchConfiguration.numberBlocks
        this.numberThreads = launchConfiguration.numberThreadsPerBlock
        this.numberIterations[0] = launchConfiguration.numberIterations

        val totalParameters = this.numberParameters * this.parameterSize
        allocateDeviceFloatMemory(this.deviceFirstMomentEstimate, totalParameters)
        allocateDeviceFloatMemory(this.deviceSecondMomentEstimate, totalParameters)
    }

    override fun launchKernel(
        numberParameters: Int,
        pointerToParameterIndices: Pointer,
        pointerToCounts : Pointer,
        pointerToParameters: Pointer,
        pointerToGradient: Pointer) : Int {
        this.step[0] += 1.0f;

        val parameters = Pointer.to(
            this.pointerToNumberIterations,
            pointerToParameterIndices,
            pointerToCounts,
            this.pointerToParameterSize,
            pointerToParameters,
            pointerToGradient,
            this.pointerToLearningRate,
            this.pointerToFirstMomentDecay,
            this.pointerToOneMinusFirstMomentDecay,
            this.pointerToSecondMomentDecay,
            this.pointerToOneMinusSecondMomentDecay,
            this.pointerToEpsilon,
            this.pointerToStep,
            this.pointerToFirstMomentEstimate,
            this.pointerToSecondMomentEstimate
        )

        val resultCode = this.kernel!!.launch(
            parameters,
            numberParameters,
            this.numberBlocks,
            this.numberThreads,
            0
        )

        return resultCode
    }

    override fun release() {
        super.release()

        this.kernel!!.destroy()

        cudaFree(this.deviceFirstMomentEstimate)
        cudaFree(this.deviceSecondMomentEstimate)
    }

}