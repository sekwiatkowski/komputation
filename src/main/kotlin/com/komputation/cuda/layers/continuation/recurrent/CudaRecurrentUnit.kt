package com.komputation.cuda.layers.continuation.recurrent

import com.komputation.cuda.allocateDeviceFloatMemory
import com.komputation.cuda.computeDeviceFloatArraySize
import com.komputation.cuda.getFloatArray
import com.komputation.cuda.kernels.Kernel
import com.komputation.cuda.kernels.launch.computeColumnwiseLaunchConfiguration
import com.komputation.cuda.layers.continuation.BasePrimitiveCudaContinuation
import com.komputation.cuda.optimization.BaseCudaUpdateRule
import com.komputation.cuda.setFloatArray
import com.komputation.instructions.continuation.activation.RecurrentActivation
import com.komputation.optimization.Optimizable
import jcuda.Pointer
import jcuda.runtime.JCuda.cudaFree

class CudaRecurrentUnit(
    name : String?,
    hiddenDimension : Int,
    maximumLength : Int,
    private var previousStateWeights : FloatArray,
    private val previousStateUpdateRule: BaseCudaUpdateRule?,
    private val activation : RecurrentActivation,
    private val createForwardKernel: () -> Kernel,
    private val createBackwardKernel: () -> Kernel,
    private val createSumKernel: () -> Kernel,
    private val maximumNumberThreadsPerBlock : Int) : BasePrimitiveCudaContinuation(name, hiddenDimension, hiddenDimension, maximumLength, maximumLength), Optimizable {

    private var forwardKernel : Kernel? = null
    private var backwardKernel : Kernel? = null
    private var sumKernel : Kernel? = null

    private val devicePreviousStateWeights = Pointer()
    private val pointerToPreviousStateWeights = Pointer.to(this.devicePreviousStateWeights)
    private val numberPreviousStateWeightEntries = this.previousStateWeights.size

    private val deviceBackwardWeightedPreviousStateWrtWeightsAccumulation = Pointer()
    private val pointerToBackwardWeightedPreviousStateWrtWeightsAccumulation = Pointer.to(this.deviceBackwardWeightedPreviousStateWrtWeightsAccumulation)

    private val deviceBackwardWeightedPreviousStateWrtWeights = Pointer()
    private val pointerToBackwardWeightedPreviousStateWrtWeights = Pointer.to(this.deviceBackwardWeightedPreviousStateWrtWeights)

    private val devicePreActivation = Pointer()
    private val pointerToPreActivation = Pointer.to(this.devicePreActivation)

    private val pointerToActivationFunction = Pointer.to(intArrayOf(this.activation.id))

    private val pointerToHiddenDimension = Pointer.to(intArrayOf(hiddenDimension))

    private val squaredHiddenDimensions = hiddenDimension * hiddenDimension
    private val pointerToSquaredHiddenDimension = Pointer.to(intArrayOf(this.squaredHiddenDimensions))

    private val propagationLaunchConfiguration = computeColumnwiseLaunchConfiguration(hiddenDimension, 1, this.maximumNumberThreadsPerBlock)
    private val propagationNumberThreads = this.propagationLaunchConfiguration.numberThreadsPerBlock
    private val pointerToPropagationIterations = Pointer.to(intArrayOf(this.propagationLaunchConfiguration.numberIterations))
    private val propagationSharedMemorySize = computeDeviceFloatArraySize(2 * hiddenDimension).toInt()

    private val sumLaunchConfigurations = computeColumnwiseLaunchConfiguration(this.squaredHiddenDimensions, 1, this.maximumNumberThreadsPerBlock)
    private val sumNumberThreads = this.sumLaunchConfigurations.numberThreadsPerBlock
    private val pointerToSumIterations = Pointer.to(intArrayOf(this.sumLaunchConfigurations.numberIterations))

    private var pointerToLengths = Pointer()

    override var deviceForwardLengths = Pointer()

    override fun acquire(maximumBatchSize: Int) {
        super.acquire(maximumBatchSize)

        allocateDeviceFloatMemory(this.devicePreActivation, this.forwardResultSize)
        allocateDeviceFloatMemory(this.deviceBackwardWeightedPreviousStateWrtWeightsAccumulation, maximumBatchSize * this.numberPreviousStateWeightEntries)
        allocateDeviceFloatMemory(this.deviceBackwardWeightedPreviousStateWrtWeights, this.numberPreviousStateWeightEntries)

        setFloatArray(this.previousStateWeights, this.numberPreviousStateWeightEntries, devicePreviousStateWeights)

        this.forwardKernel = this.createForwardKernel()
        this.backwardKernel = this.createBackwardKernel()
        this.sumKernel = this.createSumKernel()
    }

    override fun release() {
        super.release()

        cudaFree(this.devicePreActivation)
        cudaFree(this.deviceBackwardWeightedPreviousStateWrtWeightsAccumulation)
        cudaFree(this.deviceBackwardWeightedPreviousStateWrtWeights)

        this.previousStateWeights = getFloatArray(this.devicePreviousStateWeights, this.numberPreviousStateWeightEntries)

        this.forwardKernel!!.destroy()
        this.backwardKernel!!.destroy()
        this.sumKernel!!.destroy()
    }

    override fun computeForwardResult(batchSize: Int, deviceInput: Pointer, deviceInputLengths: Pointer, isTraining: Boolean) {

        this.pointerToLengths = Pointer.to(this.deviceForwardLengths)

        this.forwardKernel!!.launch(
            Pointer.to(
                this.pointerToActivationFunction,
                this.pointerToMaximumInputEntries,
                this.pointerToHiddenDimension,
                this.pointerToPropagationIterations,
                Pointer.to(deviceInput),
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

    }

    override fun computeOutputLengths(deviceInputLengths: Pointer) {
        this.deviceForwardLengths = deviceInputLengths
    }

    override fun computeBackwardResult(batchSize: Int, chain: Pointer) {

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

    }

    override fun optimize(batchSize: Int) {

        this.previousStateUpdateRule?.denseUpdate(
            batchSize,
            this.pointerToPreviousStateWeights,
            this.pointerToBackwardWeightedPreviousStateWrtWeights)
    }
}