package com.komputation.cuda.optimization.adaptive

import com.komputation.cuda.allocateDeviceFloatMemory
import com.komputation.cuda.kernels.Kernel
import com.komputation.cuda.kernels.launch.computeEntrywiseLaunchConfiguration
import com.komputation.cuda.optimization.BaseCudaUpdateRule
import com.komputation.instructions.Resourceful
import jcuda.Pointer
import jcuda.runtime.JCuda.cudaFree

class CudaAdagrad internal constructor(
    private val numberParameters : Int,
    private val dimension: Int,
    private val learningRate: Float,
    private val epsilon : Float,
    private val createKernel: () -> Kernel,
    private val numberMultiprocessors : Int,
    private val numberResidentWarps : Int,
    private val warpSize : Int,
    private val maximumNumberThreads : Int) : BaseCudaUpdateRule(), Resourceful {

    private val pointerToDimension = Pointer.to(intArrayOf(this.dimension))
    private val pointerToLearningRate = Pointer.to(floatArrayOf(this.learningRate))
    private val pointerToEpsilon = Pointer.to(floatArrayOf(this.epsilon))

    private val deviceHistory = Pointer()
    private val pointerToHistory = Pointer.to(this.deviceHistory)

    private var kernel : Kernel? = null
    private var numberBlocks = -1
    private var numberThreads = -1
    private val numberIterations = intArrayOf(-1)
    private var pointerToNumberIterations = Pointer.to(this.numberIterations)

    override fun acquire(maximumBatchSize : Int) {
        super.acquire(maximumBatchSize)

        this.kernel = this.createKernel()

        val launchConfiguration = computeEntrywiseLaunchConfiguration(this.dimension, this.numberMultiprocessors, this.numberResidentWarps, this.warpSize, this.maximumNumberThreads)
        this.numberBlocks = launchConfiguration.numberBlocks
        this.numberThreads = launchConfiguration.numberThreadsPerBlock
        this.numberIterations[0] = launchConfiguration.numberIterations

        allocateDeviceFloatMemory(this.deviceHistory, this.numberParameters * this.dimension)
    }

    override fun launchKernel(
        numberParameters: Int,
        pointerToParameterIndices: Pointer,
        pointerToCounts : Pointer,
        pointerToParameters: Pointer,
        pointerToGradient: Pointer) : Int {
        val parameters = Pointer.to(
            this.pointerToNumberIterations,
            pointerToParameterIndices,
            pointerToCounts,
            this.pointerToDimension,
            pointerToParameters,
            pointerToGradient,
            this.pointerToLearningRate,
            this.pointerToHistory,
            this.pointerToEpsilon
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

        cudaFree(this.deviceHistory)
    }

}