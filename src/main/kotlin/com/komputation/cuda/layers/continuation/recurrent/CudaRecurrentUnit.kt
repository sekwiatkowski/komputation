package com.komputation.cuda.layers.continuation.recurrent

import com.komputation.cuda.*
import com.komputation.cuda.kernels.Kernel
import com.komputation.cuda.kernels.launch.computeColumnwiseLaunchConfiguration
import com.komputation.cuda.layers.continuation.BasePrimitiveCudaContinuation
import com.komputation.cuda.optimization.BaseCudaUpdateRule
import com.komputation.instructions.continuation.activation.RecurrentActivation
import com.komputation.instructions.recurrent.ResultExtraction
import com.komputation.optimization.Optimizable
import jcuda.Pointer
import jcuda.runtime.JCuda.cudaFree

class CudaRecurrentUnit(
    name : String?,
    private val hiddenDimension : Int,
    maximumLength : Int,
    private val resultExtraction: ResultExtraction,
    private var previousStateWeights : FloatArray,
    private val previousStateUpdateRule: BaseCudaUpdateRule?,
    private val activation : RecurrentActivation,
    private val createForwardKernel: () -> Kernel,
    private val createBackwardKernel: () -> Kernel,
    private val createSumKernel: () -> Kernel,
    private val maximumNumberThreadsPerBlock : Int) : BasePrimitiveCudaContinuation(name, hiddenDimension, hiddenDimension, maximumLength, if(resultExtraction == ResultExtraction.AllSteps) maximumLength else 1), Optimizable {

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

    private var pointerToInputLengths = Pointer()

    override var deviceForwardLengths = Pointer()

    private val deviceHiddenStates = when (this.resultExtraction) {
        ResultExtraction.AllSteps -> this.deviceForwardResult
        ResultExtraction.LastStep -> Pointer()
    }
    private val pointerToHiddenStates = when (this.resultExtraction) {
        ResultExtraction.AllSteps -> this.pointerToForwardResult
        ResultExtraction.LastStep -> Pointer.to(this.deviceHiddenStates)
    }


    override fun acquire(maximumBatchSize: Int) {
        super.acquire(maximumBatchSize)

        allocateDeviceFloatMemory(this.devicePreActivation, this.forwardResultSize)
        allocateDeviceFloatMemory(this.deviceBackwardWeightedPreviousStateWrtWeightsAccumulation, this.maximumBatchSize * this.numberPreviousStateWeightEntries)
        allocateDeviceFloatMemory(this.deviceBackwardWeightedPreviousStateWrtWeights, this.numberPreviousStateWeightEntries)

        setFloatArray(this.previousStateWeights, this.numberPreviousStateWeightEntries, devicePreviousStateWeights)

        if (this.resultExtraction == ResultExtraction.LastStep) {
            setIntArray(IntArray(this.maximumBatchSize) { 1 }, this.maximumBatchSize, this.deviceForwardLengths)
            allocateDeviceFloatMemory(this.deviceHiddenStates, this.maximumBatchSize * this.maximumInputColumns * this.numberOutputRows)

        }

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

        if (this.resultExtraction == ResultExtraction.LastStep) {
            cudaFree(this.deviceForwardLengths)
            cudaFree(this.deviceHiddenStates)
        }

        this.forwardKernel!!.destroy()
        this.backwardKernel!!.destroy()
        this.sumKernel!!.destroy()
    }

    override fun computeForwardResult(batchSize: Int, deviceInput: Pointer, deviceInputLengths: Pointer, isTraining: Boolean) {

        this.pointerToInputLengths = Pointer.to(deviceInputLengths)

        val parameters =
            when (this.resultExtraction) {
                ResultExtraction.AllSteps ->  Pointer.to(
                    this.pointerToActivationFunction,
                    this.pointerToMaximumInputEntries,
                    this.pointerToHiddenDimension,
                    this.pointerToPropagationIterations,
                    Pointer.to(deviceInput),
                    this.pointerToPreActivation,
                    this.pointerToPreviousStateWeights,
                    this.pointerToInputLengths,
                    this.pointerToMaximumInputColumns,
                    this.pointerToForwardResult
                )
                ResultExtraction.LastStep ->  Pointer.to(
                    this.pointerToActivationFunction,
                    this.pointerToMaximumInputEntries,
                    this.pointerToHiddenDimension,
                    this.pointerToPropagationIterations,
                    Pointer.to(deviceInput),
                    this.pointerToPreActivation,
                    this.pointerToPreviousStateWeights,
                    this.pointerToInputLengths,
                    this.pointerToMaximumInputColumns,
                    this.pointerToHiddenStates,
                    this.pointerToForwardResult
                )
            }

        this.forwardKernel!!.launch(
            parameters,
            batchSize,
            1,
            this.propagationNumberThreads,
            this.propagationSharedMemorySize)

    }

    override fun computeOutputLengths(deviceInputLengths: Pointer) {
        if (this.resultExtraction.equals(ResultExtraction.AllSteps)) {
            this.deviceForwardLengths = deviceInputLengths
        }
    }

    override fun computeBackwardResult(batchSize: Int, chain: Pointer) {

        this.backwardKernel!!.launch(
            Pointer.to(
                this.pointerToActivationFunction,
                this.pointerToMaximumInputEntries,
                this.pointerToHiddenDimension,
                this.pointerToSquaredHiddenDimension,
                this.pointerToInputLengths,
                this.pointerToPropagationIterations,
                this.pointerToHiddenStates,
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