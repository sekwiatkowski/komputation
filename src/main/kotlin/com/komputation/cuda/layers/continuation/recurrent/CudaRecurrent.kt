package com.komputation.cuda.layers.continuation.recurrent

import com.komputation.cuda.allocateDeviceFloatMemory
import com.komputation.cuda.computeDeviceFloatArraySize
import com.komputation.cuda.getFloatArray
import com.komputation.cuda.kernels.Kernel
import com.komputation.cuda.kernels.launch.computeColumnwiseLaunchConfiguration
import com.komputation.cuda.layers.continuation.BaseCudaHigherOrderContinuation
import com.komputation.cuda.layers.continuation.projection.CublasProjection
import com.komputation.cuda.optimization.BaseCudaUpdateRule
import com.komputation.cuda.setFloatArray
import com.komputation.instructions.continuation.activation.RecurrentActivation
import com.komputation.optimization.Optimizable
import jcuda.Pointer
import jcuda.runtime.JCuda.cudaFree

class CudaRecurrent(
    name : String?,
    private val hiddenDimension : Int,
    private val inputProjection : CublasProjection,
    private var previousStateWeights : FloatArray,
    private val previousStateUpdateRule: BaseCudaUpdateRule?,
    private val activation : RecurrentActivation,
    private val createForwardKernel: () -> Kernel,
    private val createBackwardKernel: () -> Kernel,
    private val createSumKernel: () -> Kernel,
    private val maximumNumberThreadsPerBlock : Int) : BaseCudaHigherOrderContinuation(name, inputProjection, inputProjection), Optimizable {

    override val deviceForwardResult = Pointer()
    private val pointerToForwardResult = Pointer.to(this.deviceForwardResult)
    override val deviceBackwardResult = Pointer()
    private val pointerToBackwardResult = Pointer.to(this.deviceBackwardResult)

    private var forwardKernel : Kernel? = null
    private var backwardKernel : Kernel? = null
    private var sumKernel : Kernel? = null

    private val devicePreviousStateWeights = Pointer()

    private val deviceBackwardWeightedPreviousStateWrtWeightsAccumulation = Pointer()
    private val pointerToBackwardWeightedPreviousStateWrtWeightsAccumulation = Pointer.to(this.deviceBackwardWeightedPreviousStateWrtWeightsAccumulation)

    private val deviceBackwardWeightedPreviousStateWrtWeights = Pointer()
    private val pointerToBackwardWeightedPreviousStateWrtWeights = Pointer.to(this.deviceBackwardWeightedPreviousStateWrtWeights)

    private val devicePreActivation = Pointer()
    private val pointerToPreActivation = Pointer.to(this.devicePreActivation)

    private val numberPreviousStateWeightEntries = this.previousStateWeights.size

    override fun acquire(maximumBatchSize: Int) {
        super.acquire(maximumBatchSize)

        allocateDeviceFloatMemory(this.deviceForwardResult, this.forwardResultSize)
        allocateDeviceFloatMemory(this.devicePreActivation, this.forwardResultSize)
        allocateDeviceFloatMemory(this.deviceBackwardResult, this.backwardResultSize)
        allocateDeviceFloatMemory(this.deviceBackwardWeightedPreviousStateWrtWeightsAccumulation, maximumBatchSize * this.numberPreviousStateWeightEntries)
        allocateDeviceFloatMemory(this.deviceBackwardWeightedPreviousStateWrtWeights, this.numberPreviousStateWeightEntries)

        setFloatArray(this.previousStateWeights, this.numberPreviousStateWeightEntries, devicePreviousStateWeights)

        this.forwardKernel = this.createForwardKernel()
        this.backwardKernel = this.createBackwardKernel()
        this.sumKernel = this.createSumKernel()
    }

    override fun release() {
        super.release()

        cudaFree(this.deviceForwardResult)
        cudaFree(this.deviceBackwardResult)
        cudaFree(this.devicePreActivation)
        cudaFree(this.deviceBackwardWeightedPreviousStateWrtWeightsAccumulation)
        cudaFree(this.deviceBackwardWeightedPreviousStateWrtWeights)

        this.previousStateWeights = getFloatArray(this.devicePreviousStateWeights, this.numberPreviousStateWeightEntries)

        this.forwardKernel!!.destroy()
        this.backwardKernel!!.destroy()
        this.sumKernel!!.destroy()
    }

    private val pointerToActivationFunction = Pointer.to(intArrayOf(this.activation.id))

    private val pointerToHiddenDimension = Pointer.to(intArrayOf(this.hiddenDimension))

    private val squaredHiddenDimensions = this.hiddenDimension * this.hiddenDimension
    private val pointerToSquaredHiddenDimension = Pointer.to(intArrayOf(this.squaredHiddenDimensions))

    private val propagationLaunchConfiguration = computeColumnwiseLaunchConfiguration(this.hiddenDimension, 1, this.maximumNumberThreadsPerBlock)
    private val propagationNumberThreads = this.propagationLaunchConfiguration.numberThreadsPerBlock
    private val pointerToPropagationIterations = Pointer.to(intArrayOf(this.propagationLaunchConfiguration.numberIterations))
    private val propagationSharedMemorySize = computeDeviceFloatArraySize(2 * this.hiddenDimension).toInt()

    private val sumLaunchConfigurations = computeColumnwiseLaunchConfiguration(this.squaredHiddenDimensions, 1, this.maximumNumberThreadsPerBlock)
    private val sumNumberThreads = this.sumLaunchConfigurations.numberThreadsPerBlock
    private val pointerToSumIterations = Pointer.to(intArrayOf(this.sumLaunchConfigurations.numberIterations))

    private val pointerToPreviousStateWeights = Pointer.to(this.devicePreviousStateWeights)

    private var pointerToLengths = Pointer()

    private val batchSizeArray = intArrayOf(0)
    private val pointerToBatchSize = Pointer.to(this.batchSizeArray)

    override fun forward(batchSize: Int, deviceInput: Pointer, deviceInputLengths: Pointer, isTraining: Boolean): Pointer {

        this.batchSizeArray[0] = batchSize

        /*
            Project the input:
               a1 b1 NaN NaN | c1 d1 e1 f1
               a2 b1 NaN NaN | c2 d2 e2 f2
               a2 b1 NaN NaN | c3 d3 e3 f3
            w1
            w2
         */
        val deviceProjectedInput = this.inputProjection.forward(batchSize, deviceInput, deviceInputLengths, isTraining)

        this.pointerToLengths = Pointer.to(this.deviceForwardLengths)

        this.forwardKernel!!.launch(
            Pointer.to(
                this.pointerToActivationFunction,
                this.pointerToMaximumInputEntries,
                this.pointerToHiddenDimension,
                this.pointerToPropagationIterations,
                Pointer.to(deviceProjectedInput),
                this.pointerToPreActivation,
                this.pointerToPreviousStateWeights,
                this.pointerToLengths,
                this.pointerToMaximumInputColumns,
                this.pointerToForwardResult
            ),
            batchSize,
            1,
            this.propagationNumberThreads,
            this.propagationSharedMemorySize)

        return this.deviceForwardResult
    }

    override fun backward(batchSize: Int, chain: Pointer): Pointer {

        this.backwardKernel!!.launch(
            Pointer.to(
                this.pointerToActivationFunction,
                this.pointerToMaximumInputEntries,
                this.pointerToHiddenDimension,
                this.pointerToSquaredHiddenDimension,
                this.pointerToLengths,
                this.pointerToPropagationIterations,
                this.pointerToForwardResult,
                this.pointerToPreActivation,
                this.pointerToPreviousStateWeights,
                this.pointerToBackwardWeightedPreviousStateWrtWeightsAccumulation,
                Pointer.to(chain),
                this.pointerToBackwardResult
            ),
            batchSize,
            1,
            this.propagationNumberThreads,
            this.propagationSharedMemorySize
        )

        this.sumKernel!!.launch(
            Pointer.to(
                this.pointerToBackwardWeightedPreviousStateWrtWeightsAccumulation,
                this.pointerToBackwardWeightedPreviousStateWrtWeights,
                this.pointerToBatchSize,
                this.pointerToSquaredHiddenDimension,
                this.pointerToSumIterations
            ),
            1,
            1,
            this.sumNumberThreads,
            0
        )

        this.inputProjection.backward(batchSize, this.deviceBackwardResult)

        return this.inputProjection.deviceBackwardResult

    }

    override fun optimize(batchSize: Int) {
        this.inputProjection.optimize(batchSize)

        this.previousStateUpdateRule?.denseUpdate(
            batchSize,
            this.pointerToPreviousStateWeights,
            this.pointerToBackwardWeightedPreviousStateWrtWeights)
    }
}