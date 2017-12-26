package com.komputation.cpu.network

import com.komputation.cpu.instructions.CpuContinuationInstruction
import com.komputation.cpu.instructions.CpuEntryPointInstruction
import com.komputation.cpu.instructions.CpuLossFunctionInstruction
import com.komputation.cpu.layers.CpuContinuation
import com.komputation.cpu.layers.CpuEntryPoint
import com.komputation.cpu.workflow.CpuBinaryTester
import com.komputation.cpu.workflow.CpuMulticlassTester
import com.komputation.cpu.workflow.CpuTester
import com.komputation.cpu.workflow.CpuTrainer
import com.komputation.instructions.Resourceful
import com.komputation.instructions.acquireRecursively
import com.komputation.instructions.releaseRecursively
import com.komputation.matrix.Matrix
import com.komputation.matrix.partitionIndices
import com.komputation.optimization.Optimizable

class Network internal constructor(
    private val maximumBatchSize: Int,
    entryPointInstruction: CpuEntryPointInstruction,
    vararg continuationInstructions: CpuContinuationInstruction) {

    private val entryPoint : CpuEntryPoint
    private val layers : Array<CpuContinuation>
    private val numberPredictionRows : Int
    private val minimumNumberPredictionColumns : Int
    private val maximumNumberPredictionColumns : Int

    init {
        with(continuationInstructions[0]) {
            setInputDimensionsFromPreviousInstruction(entryPointInstruction.numberOutputRows, entryPointInstruction.minimumNumberOutputColumns, entryPointInstruction.maximumNumberOutputColumns)
        }

        (1 until continuationInstructions.size).forEach { index ->
            val previousLayer = continuationInstructions[index - 1]
            val nextLayer = continuationInstructions[index]

            nextLayer.setInputDimensionsFromPreviousInstruction(previousLayer.numberOutputRows, previousLayer.minimumNumberOutputColumns, previousLayer.maximumNumberOutputColumns)
        }

        val lastLayerInstruction = continuationInstructions.last()
        this.numberPredictionRows = lastLayerInstruction.numberOutputRows
        this.minimumNumberPredictionColumns = lastLayerInstruction.minimumNumberOutputColumns
        this.maximumNumberPredictionColumns = lastLayerInstruction.maximumNumberOutputColumns

        this.entryPoint = entryPointInstruction.buildForCpu()
        this.layers = Array(continuationInstructions.size) { index -> continuationInstructions[index].buildForCpu() }
    }

    private val optimizables = listOf(this.entryPoint).plus(this.layers).filterIsInstance(Optimizable::class.java).reversed().toTypedArray()

    private val forwardPropagator = CpuForwardPropagator(this.entryPoint, this.layers)
    private val backwardPropagator = CpuBackwardPropagator(this.entryPoint, this.layers)

    fun getEntryPoint() =
        this.entryPoint

    fun getLayer(index : Int) =
        this.layers[index]

    fun training(
        inputs: Array<Matrix>,
        targets: Array<FloatArray>,
        numberIterations : Int,
        loss: CpuLossFunctionInstruction,
        afterEachIteration : ((index : Int, loss : Float) -> Unit)? = null) : CpuTrainer {

        loss.setInputDimensionsFromPreviousInstruction(this.numberPredictionRows, this.minimumNumberPredictionColumns, this.maximumNumberPredictionColumns)

        return CpuTrainer(
            this.forwardPropagator,
            this.backwardPropagator,
            this.optimizables,
            inputs,
            targets,
            numberIterations,
            this.maximumBatchSize,
            loss.buildForCpu(),
            afterEachIteration)
    }

    fun test(
        inputs: Array<Matrix>,
        targets: Array<FloatArray>,
        batchSize: Int,
        numberCategories : Int,
        length : Int = 1) : CpuTester {
        val batches = partitionIndices(inputs.size, batchSize)

        val tester = if (numberCategories == 1) {
            CpuBinaryTester(length)
        }
        else {
            CpuMulticlassTester(numberCategories, length)
        }

        return CpuTester(this.forwardPropagator, batches, inputs, targets, tester)
    }

    init {
        acquireRecursively(this.entryPoint, this.maximumBatchSize)

        for (layer in this.layers) {
            acquireRecursively(layer, this.maximumBatchSize)
        }
    }

    fun free() {
        for (layer in this.layers) {
            releaseRecursively(layer, CpuContinuation::class.java)
        }

        if (this.entryPoint is Resourceful) {
            this.entryPoint.release()
        }
    }

    fun predict(input : Matrix) =
        this.forwardPropagator.forward(0, input, false).forwardResult.copyOf()

}

fun network(
    maximumBatchSize: Int,
    entryPointInstruction: CpuEntryPointInstruction,
    vararg continuationInstructions: CpuContinuationInstruction)  =
    Network(maximumBatchSize, entryPointInstruction, *continuationInstructions)