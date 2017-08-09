package shape.komputation.cpu

import shape.komputation.cpu.layers.ForwardLayerState
import shape.komputation.layers.CpuEntryPointInstruction
import shape.komputation.layers.CpuForwardLayerInstruction
import shape.komputation.layers.Resourceful
import shape.komputation.loss.CpuLossFunctionInstruction
import shape.komputation.matrix.Matrix
import shape.komputation.matrix.partitionIndices
import shape.komputation.optimization.Optimizable

val printLoss = { _ : Int, loss : Float -> println(loss) }

class Network(entryPointInstruction: CpuEntryPointInstruction, vararg forwardLayerInstructions: CpuForwardLayerInstruction) {

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

        return previousLayerState.forwardResult

    }

    fun backward(withinBatch: Int, lossGradient: FloatArray) {

        var chain = lossGradient

        for(indexLayer in this.numberLayers - 1 downTo 0) {

            val layer = this.layers[indexLayer]

            chain = layer.backward(withinBatch, chain)

        }

        this.entryPoint.backward(chain)

    }

    fun optimize(scalingFactor : Float) {

        for (optimizable in this.optimizables) {

            optimizable.optimize(scalingFactor)

        }

    }

    fun train(
        inputs: Array<Matrix>,
        targets: Array<FloatArray>,
        loss: CpuLossFunctionInstruction,
        numberIterations : Int,
        maximumBatchSize: Int,
        afterEachIteration : ((index : Int, loss : Float) -> Unit)? = null): Long {

        val lossFunction = loss.buildForCpu()

        val numberExamples = inputs.size

        val batches = partitionIndices(numberExamples, maximumBatchSize)

        this.acquireLayerResources(maximumBatchSize)

        if (lossFunction is Resourceful) {

            lossFunction.acquire(maximumBatchSize)

        }

        val start = System.currentTimeMillis()

        repeat(numberIterations) { indexIteration ->

            var iterationLoss = 0.0f

            for (batch in batches) {

                var batchLoss = 0.0f

                for ((withinBatch, indexExample) in batch.withIndex()) {

                    val input = inputs[indexExample]
                    val target = targets[indexExample]

                    val prediction = this.forward(withinBatch, input, true)

                    val instanceLoss = lossFunction.forward(prediction, target)

                    lossFunction.backward(prediction, target)

                    this.backward(withinBatch, lossFunction.backwardResult)

                    batchLoss += instanceLoss

                }

                iterationLoss += batchLoss

                val scalingFactor = 1.0f.div(batch.size.toFloat())

                this.optimize(scalingFactor)

            }

            if (afterEachIteration != null) {

                afterEachIteration(indexIteration, iterationLoss)

            }

        }

        val stop = System.currentTimeMillis()

        val time = stop - start

        this.releaseLayerResources()

        return time

    }

    fun test(
        inputs: Array<Matrix>,
        targets: Array<FloatArray>,
        batchSize: Int,
        isCorrect: (FloatArray, FloatArray) -> Boolean) : BooleanArray {

        val numberInstances = inputs.size

        val batches = partitionIndices(numberInstances, batchSize)

        val results = BooleanArray(numberInstances)

        for((withinBatch, batch) in batches.withIndex()) {

            for (index in batch) {

                val input = inputs[index]
                val target = targets[index]

                val prediction = this.forward(withinBatch, input, false)

                results[index] = isCorrect(prediction, target)

            }

        }

        return results

    }

    fun acquireLayerResources(maximumBatchSize: Int) {

        if (this.entryPoint is Resourceful) {

            this.entryPoint.acquire(maximumBatchSize)

        }

        for (layer in this.layers) {

            if (layer is Resourceful) {

                layer.acquire(maximumBatchSize)

            }

        }

    }

    fun releaseLayerResources() {

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