package com.komputation.cpu.network

import com.komputation.cpu.layers.CpuForwardLayer
import com.komputation.cpu.workflow.CpuBinaryTester
import com.komputation.cpu.workflow.CpuMulticlassTester
import com.komputation.cpu.workflow.CpuTester
import com.komputation.cpu.workflow.CpuTrainer
import com.komputation.layers.*
import com.komputation.loss.CpuLossFunctionInstruction
import com.komputation.matrix.Matrix
import com.komputation.matrix.partitionIndices
import com.komputation.optimization.Optimizable

class Network(
    private val maximumBatchSize: Int,
    entryPointInstruction: CpuEntryPointInstruction,
    vararg forwardLayerInstructions: CpuForwardLayerInstruction) {

    private val entryPoint = entryPointInstruction.buildForCpu()
    private val layers = Array(forwardLayerInstructions.size) { index -> forwardLayerInstructions[index].buildForCpu() }
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
        afterEachIteration : ((index : Int, loss : Float) -> Unit)? = null) =

        CpuTrainer(
            this.forwardPropagator,
            this.backwardPropagator,
            this.optimizables,
            inputs,
            targets,
            numberIterations,
            this.maximumBatchSize,
            loss.buildForCpu(),
            afterEachIteration)

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
            releaseRecursively(layer, CpuForwardLayer::class.java)
        }

        if (this.entryPoint is Resourceful) {
            this.entryPoint.release()
        }
    }

    fun predict(input : Matrix) =
        this.forwardPropagator.forward(0, input, false).forwardResult.copyOf()

}