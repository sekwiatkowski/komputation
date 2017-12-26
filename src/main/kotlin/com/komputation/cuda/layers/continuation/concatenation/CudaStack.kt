package com.komputation.cuda.layers.continuation.concatenation

import com.komputation.cuda.kernels.Kernel
import com.komputation.cuda.kernels.launch.KernelLaunchConfiguration
import com.komputation.cuda.kernels.launch.computeColumnwiseLaunchConfiguration
import com.komputation.cuda.layers.CudaContinuation
import com.komputation.cuda.layers.continuation.BaseCudaFixedNumberColumnsContinuation
import jcuda.Pointer

class CudaStack(
    name : String?,
    numberInputRows : Int,
    numbersOutputRows : IntArray,
    maximumInputColumns : Int,
    maximumOutputColumns : Int,
    private val createForwardKernel: () -> Kernel,
    private val createBackwardKernel: () -> Kernel,
    private val layers: Array<CudaContinuation>,
    maximumNumberThreads : Int) : BaseCudaFixedNumberColumnsContinuation(name, numberInputRows, numbersOutputRows.sum(), maximumInputColumns) {

    private var forwardKernel : Kernel? = null
    private var backwardKernel : Kernel? = null

    override fun acquire(maximumBatchSize: Int) {
        super.acquire(maximumBatchSize)

        this.forwardKernel = this.createForwardKernel()
        this.backwardKernel = this.createBackwardKernel()
    }

    private val numberLayers = this.layers.size
    private val pointersToFirstRowInDestination: Array<Pointer>
    private val pointersToSourceMaximumEntries: Array<Pointer>
    private val pointersToSourceNumberRows: Array<Pointer>
    private val pointersToNumberIterations: Array<Pointer>
    private val forwardLaunchConfigurations : Array<KernelLaunchConfiguration>

    init {
        val firstRowsInDestination = IntArray(this.numberLayers)

        numbersOutputRows.foldIndexed(0) { index, oldTotal, numberOutputRows ->
            firstRowsInDestination[index] = oldTotal
            oldTotal.plus(numberOutputRows)
        }

        this.pointersToFirstRowInDestination = Array(this.numberLayers) { index -> Pointer.to(intArrayOf(firstRowsInDestination[index])) }

        val sourceMaximumEntries = Array(this.numberLayers) { index ->
            numbersOutputRows[index] * maximumOutputColumns
        }

        this.pointersToSourceMaximumEntries = Array(this.numberLayers) { index -> Pointer.to(intArrayOf(sourceMaximumEntries[index])) }
        this.pointersToSourceNumberRows = Array(this.numberLayers) { index -> Pointer.to(intArrayOf(numbersOutputRows[index])) }

        this.forwardLaunchConfigurations = Array(this.numberLayers) { index ->
            computeColumnwiseLaunchConfiguration(numbersOutputRows[index], maximumOutputColumns, maximumNumberThreads)
        }

        this.pointersToNumberIterations = Array(this.numberLayers) { index -> Pointer.to(intArrayOf(this.forwardLaunchConfigurations[index].numberIterations)) }
    }

    override fun computeForwardResult(batchSize: Int, deviceInput: Pointer, deviceInputLengths: Pointer, batchMaximumInputLength: Int, isTraining: Boolean) {
        for ((index, layer) in (this.layers.withIndex())) {
            val source = layer.forward(batchSize, deviceInput, deviceInputLengths, batchMaximumInputLength, isTraining)

            val launchConfiguration = this.forwardLaunchConfigurations[index]

            this.forwardKernel!!.launch(
                Pointer.to(
                    this.pointersToSourceMaximumEntries[index],
                    this.pointersToSourceNumberRows[index],
                    this.pointerToMaximumOutputEntries,
                    this.pointerToNumberOutputRows,
                    this.pointersToFirstRowInDestination[index],
                    Pointer.to(source),
                    this.pointerToForwardResult,
                    this.pointersToNumberIterations[index]
                ),
                batchSize,
                launchConfiguration.numberBlocks,
                launchConfiguration.numberThreadsPerBlock,
                0
            )
        }
    }

    override fun computeBackwardResult(batchSize: Int, chain: Pointer) {
        TODO("Not implemented yet")
        /* this.backwardKernel!!.launch(
            Pointer.to(
            ),
            1,
            1,
            1,
            0
        ) */
    }

    override fun release() {
        super.release()

        this.forwardKernel!!.destroy()
        this.backwardKernel!!.destroy()
    }

}