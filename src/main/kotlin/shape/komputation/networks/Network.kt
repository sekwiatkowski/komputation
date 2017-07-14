package shape.komputation.networks

import shape.komputation.cpu.loss.CpuLossFunction
import shape.komputation.layers.CpuEntryPointInstruction
import shape.komputation.layers.CpuForwardLayerInstruction
import shape.komputation.matrix.DoubleMatrix
import shape.komputation.matrix.Matrix
import shape.komputation.matrix.partitionIndices
import shape.komputation.optimization.Optimizable

val printLoss = { _ : Int, loss : Double -> println(loss) }

class Network(entryPointInstruction: CpuEntryPointInstruction, vararg forwardLayerInstructions: CpuForwardLayerInstruction) {

    private val entryPoint = entryPointInstruction.buildForCpu()

    private val layers = forwardLayerInstructions.map { it.buildForCpu() }
    private val numberLayers = this.layers.size

    private val optimizables = listOf(this.entryPoint).plus(this.layers).filterIsInstance(Optimizable::class.java).reversed()

    fun forward(input : Matrix, isTraining : Boolean) : DoubleMatrix {

        var output = this.entryPoint.forward(input)

        for (layer in this.layers) {

            output = layer.forward(output, isTraining)

        }

        return output

    }

    fun backward(lossGradient : DoubleMatrix): DoubleMatrix {

        var chain = lossGradient

        for(indexLayer in this.numberLayers - 1 downTo 0) {

            val layer = this.layers[indexLayer]

            chain = layer.backward(chain)

        }

        return this.entryPoint.backward(chain)

    }

    fun optimize(scalingFactor : Double) {

        for (optimizable in this.optimizables) {

            optimizable.optimize(scalingFactor)

        }

    }

    fun train(
        inputs: Array<Matrix>,
        targets: Array<DoubleMatrix>,
        lossFunction: CpuLossFunction,
        numberIterations : Int,
        batchSize : Int,
        afterEachIteration : ((index : Int, loss : Double) -> Unit)? = null) {

        val numberExamples = inputs.size

        val batches = partitionIndices(numberExamples, batchSize)

        repeat(numberIterations) { indexIteration ->

            var iterationLoss = 0.0

            for (batch in batches) {

                var batchLoss = 0.0

                for (indexExample in batch) {

                    val input = inputs[indexExample]
                    val target = targets[indexExample]

                    val prediction = this.forward(input, true)

                    val loss = lossFunction.forward(prediction, target)

                    val lossGradient = lossFunction.backward(prediction, target)

                    this.backward(lossGradient)

                    batchLoss += loss

                }

                iterationLoss += batchLoss

                val scalingFactor = 1.0.div(batch.size.toDouble())

                this.optimize(scalingFactor)

            }

            if (afterEachIteration != null) {

                afterEachIteration(indexIteration, iterationLoss)

            }

        }

    }

    fun test(
        inputs: Array<Matrix>,
        targets: Array<DoubleMatrix>,
        isCorrect: (DoubleMatrix, DoubleMatrix) -> Boolean) : BooleanArray {

        val size = inputs.size

        val results = BooleanArray(size)

        for(index in 0..size -1) {

            val input = inputs[index]
            val target = targets[index]

            val prediction = this.forward(input, false)

            results[index] = isCorrect(prediction, target)

        }

        return results

    }

}