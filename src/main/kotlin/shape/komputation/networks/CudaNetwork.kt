package shape.komputation.networks

import jcuda.Pointer
import jcuda.runtime.JCuda.cudaFree
import shape.komputation.cuda.allocateDeviceMemory
import shape.komputation.cuda.loss.CudaLossFunction
import shape.komputation.cuda.setUpCudaEnvironment
import shape.komputation.cuda.setVector
import shape.komputation.layers.CudaEntryPointInstruction
import shape.komputation.layers.CudaForwardLayerInstruction
import shape.komputation.layers.Resourceful
import shape.komputation.matrix.DoubleMatrix
import shape.komputation.matrix.Matrix
import shape.komputation.matrix.partitionIndices
import shape.komputation.optimization.Optimizable

class CudaNetwork(entryPointInstruction: CudaEntryPointInstruction, vararg forwardLayerInstructions: CudaForwardLayerInstruction) {

    private val cudaEnvironment = setUpCudaEnvironment()

    private val entryPoint = entryPointInstruction.buildForCuda()

    private val layers = forwardLayerInstructions.map { it.buildForCuda(this.cudaEnvironment) }
    private val numberLayers = this.layers.size
    private val optimizables = listOf(this.entryPoint).plus(this.layers).filterIsInstance(Optimizable::class.java).reversed()

    fun forward(input : Matrix) : Pointer {

        var output = this.entryPoint.forward(input)

        for (layer in this.layers) {

            output = layer.forward(output)

        }

        return output

    }

    fun backward(lossGradient : Pointer): Pointer {

        var chain = lossGradient

        for(indexLayer in this.numberLayers - 1 downTo 0) {

            val layer = this.layers[indexLayer]

            chain = layer.backward(chain)

        }

        return this.entryPoint.backward(chain)

    }

    fun train(
        inputs: Array<Matrix>,
        targets: Array<DoubleMatrix>,
        lossFunction: CudaLossFunction,
        numberIterations : Int,
        batchSize : Int,
        afterEachIteration : ((index : Int, loss : Double) -> Unit)? = null) {

        val numberExamples = inputs.size

        val batches = partitionIndices(numberExamples, batchSize)

        this.acquireLayerResources()

        val deviceTarget = Pointer()
        val firstTarget = targets.first()
        val targetSize = firstTarget.numberRows * firstTarget.numberColumns
        allocateDeviceMemory(deviceTarget, targetSize)

        repeat(numberIterations) { indexIteration ->

            for (batch in batches) {

                var batchLoss = 0.0

                for (indexExample in batch) {

                    val input = inputs[indexExample]

                    val target = targets[indexExample]
                    setVector(deviceTarget, target.entries, targetSize)

                    val prediction = this.forward(input)

                    val loss = lossFunction.forward(prediction, deviceTarget)

                    val lossGradient = lossFunction.backward(prediction, deviceTarget)

                    this.backward(lossGradient)

                    batchLoss += loss

                }

                val scalingFactor = 1.0.div(batch.size.toDouble())

                this.optimize(scalingFactor)

            }

            if (afterEachIteration != null) {

                afterEachIteration(indexIteration, 0.0)

            }

        }

        cudaFree(deviceTarget)

        this.releaseLayerResources()


    }

    fun optimize(scalingFactor : Double) {

        for (optimizable in this.optimizables) {

            optimizable.optimize(scalingFactor)

        }

    }

    fun acquireLayerResources() {

        for (layer in this.layers) {

            if (layer is Resourceful) {

                layer.acquire()

            }

        }

    }

    fun releaseLayerResources() {

        for (layer in this.layers) {

            if (layer is Resourceful) {

                layer.release()

            }

        }

    }

}