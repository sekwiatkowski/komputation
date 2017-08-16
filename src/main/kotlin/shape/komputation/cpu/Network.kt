package shape.komputation.cpu

import shape.komputation.cpu.layers.ForwardLayerState
import shape.komputation.cpu.workflow.CpuTester
import shape.komputation.cpu.workflow.CpuTrainer
import shape.komputation.layers.CpuEntryPointInstruction
import shape.komputation.layers.CpuForwardLayerInstruction
import shape.komputation.layers.Resourceful
import shape.komputation.loss.CpuLossFunctionInstruction
import shape.komputation.matrix.Matrix
import shape.komputation.optimization.Optimizable

val printLoss = { _ : Int, loss : Float -> println(loss) }

class Network(
    private val maximumBatchSize: Int,
    entryPointInstruction: CpuEntryPointInstruction,
    vararg forwardLayerInstructions: CpuForwardLayerInstruction) {

    private val entryPoint = entryPointInstruction.buildForCpu()

    private val layers = forwardLayerInstructions.map { it.buildForCpu() }
    private val numberLayers = this.layers.size

    private val optimizables = listOf(this.entryPoint).plus(this.layers).filterIsInstance(Optimizable::class.java).reversed()

    fun forward(withinBatch : Int, input : Matrix, isTraining : Boolean) : FloatArray {

        this.entryPoint.forward(input)

        var previousLayerState : ForwardLayerState = this.entryPoint

        for (layer in this.layers) {

            layer.forward(withinBatch, previousLayerState.numberOutputColumns, previousLayerState.forwardResult, isTraining)

            previousLayerState = layer

        }

        val result = previousLayerState.forwardResult

        return result

    }

    fun backward(withinBatch: Int, lossGradient: FloatArray) : FloatArray {

        var chain = lossGradient

        for(indexLayer in this.numberLayers - 1 downTo 0) {

            val layer = this.layers[indexLayer]

            chain = layer.backward(withinBatch, chain)

        }

        val result = this.entryPoint.backward(chain)

        return result

    }

    fun optimize(scalingFactor : Float) {

        for (optimizable in this.optimizables) {

            optimizable.optimize(scalingFactor)

        }

    }

    fun training(
        inputs: Array<Matrix>,
        targets: Array<FloatArray>,
        numberIterations : Int,
        loss: CpuLossFunctionInstruction,
        afterEachIteration : ((index : Int, loss : Float) -> Unit)? = null) =

        CpuTrainer(
            this,
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
        length : Int = 1) =

        CpuTester(
            this,
            inputs,
            targets,
            batchSize,
            numberCategories,
            length
        )

    init {

        if (this.entryPoint is Resourceful) {

            this.entryPoint.acquire(this.maximumBatchSize)

        }

        for (layer in this.layers) {

            if (layer is Resourceful) {

                layer.acquire(this.maximumBatchSize)

            }

        }

    }

    fun free() {

        for (layer in this.layers) {

            if (layer is Resourceful) {

                layer.release()

            }

        }

        if (this.entryPoint is Resourceful) {

            this.entryPoint.release()

        }

    }

}