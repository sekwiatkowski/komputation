package com.komputation.cuda.layers.continuation.recurrent

import com.komputation.cpu.layers.recurrent.Direction
import com.komputation.cuda.allocateDeviceFloatMemory
import com.komputation.cuda.computeDeviceFloatArraySize
import com.komputation.cuda.kernels.Kernel
import com.komputation.cuda.kernels.launch.computeColumnwiseLaunchConfiguration
import com.komputation.cuda.layers.continuation.BaseCudaHigherOrderContinuation
import com.komputation.cuda.layers.continuation.projection.CublasProjection
import com.komputation.cuda.network.cudaNetwork
import com.komputation.initialization.uniformInitialization
import com.komputation.instructions.Resourceful
import com.komputation.instructions.continuation.activation.RecurrentActivation
import com.komputation.instructions.entry.input
import com.komputation.instructions.recurrent.ResultExtraction
import com.komputation.instructions.recurrent.recurrent
import com.komputation.matrix.floatMatrix
import jcuda.Pointer
import jcuda.runtime.JCuda.cudaFree
import java.util.*

class CudaRecurrent(
    name : String?,
    private val maximumLength : Int,
    private val hiddenDimension : Int,
    private val resultExtraction: ResultExtraction,
    private val inputProjection : CublasProjection,
    private val activation : RecurrentActivation,
    private val createForwardKernel: () -> Kernel,
    private val maximumNumberThreadsPerBlock : Int) : BaseCudaHigherOrderContinuation(name, inputProjection, inputProjection), Resourceful {

    override val deviceForwardResult = Pointer()
    private val pointerToForwardResult = Pointer.to(this.deviceForwardResult)
    override val deviceBackwardResult: Pointer
        get() = this.inputProjection.deviceBackwardResult

    private var forwardKernel : Kernel? = null

    override fun acquire(maximumBatchSize: Int) {
        super.acquire(maximumBatchSize)

        val resultSize = this.maximumBatchSize * when (resultExtraction) {
            ResultExtraction.LastStep -> this.hiddenDimension
            ResultExtraction.AllSteps -> this.maximumLength * this.hiddenDimension
        }

        allocateDeviceFloatMemory(this.deviceForwardResult, computeDeviceFloatArraySize(resultSize).toInt())

        this.forwardKernel = this.createForwardKernel()
    }

    override fun release() {
        super.release()

        cudaFree(this.deviceForwardResult)

        this.forwardKernel!!.destroy()
    }

    private val pointerToActivationFunction = Pointer.to(intArrayOf(this.activation.id))

    private val pointerToHiddenDimension = Pointer.to(intArrayOf(this.hiddenDimension))

    private val pointerToResultExtraction = Pointer.to(intArrayOf(this.resultExtraction.id))

    private val forwardLaunchConfiguration = computeColumnwiseLaunchConfiguration(this.hiddenDimension, 1, this.maximumNumberThreadsPerBlock)
    private val forwardThreads = forwardLaunchConfiguration.numberThreadsPerBlock
    private val pointerToForwardIterations = Pointer.to(intArrayOf(this.forwardLaunchConfiguration.numberIterations))
    private val forwardSharedMemorySize = computeDeviceFloatArraySize(this.hiddenDimension).toInt()

    override fun forward(batchSize: Int, deviceInput: Pointer, deviceInputLengths: Pointer, isTraining: Boolean): Pointer {

        /*
            Project the input:
               a1 b1 NaN NaN | c1 d1 e1 f1
               a2 b1 NaN NaN | c2 d2 e2 f2
               a2 b1 NaN NaN | c3 d3 e3 f3
            w1
            w2
         */
        val projectedInput = this.inputProjection.forward(batchSize, deviceInput, deviceInputLengths, isTraining)

        this.forwardKernel!!.launch(
            Pointer.to(
                this.pointerToActivationFunction,
                this.pointerToMaximumInputEntries,
                this.pointerToHiddenDimension,
                this.pointerToForwardIterations,
                Pointer.to(projectedInput),
                Pointer.to(this.inputProjection.deviceForwardLengths),
                this.pointerToResultExtraction,
                this.pointerToForwardResult
            ),
            batchSize,
            1,
            this.forwardThreads,
            this.forwardSharedMemorySize)

        return projectedInput
    }

    override fun backward(batchSize: Int, chain: Pointer): Pointer {
        TODO("not implemented")
    }

}

fun main(args: Array<String>) {

    val random = Random(1)

    val network = cudaNetwork(2, input(3, 1, 4), recurrent(2, RecurrentActivation.ReLU, ResultExtraction.AllSteps, Direction.LeftToRight, uniformInitialization(random, -0.1f, 1.0f)))

    val firstInstance = floatMatrix(
        3, 4,
        111f,
        112f,
        113f,
        121f,
        121f,
        123f,
        Float.NaN,
        Float.NaN,
        Float.NaN,
        Float.NaN,
        Float.NaN,
        Float.NaN
    )

    val secondInstance = floatMatrix(
        3, 4,
        211f,
        212f,
        213f,
        221f,
        222f,
        223f,
        231f,
        232f,
        232f,
        241f,
        242f,
        243f
    )

    network.predict(arrayOf(firstInstance, secondInstance))

    network.free()

}