package shape.komputation.cuda

import jcuda.Pointer
import jcuda.runtime.JCuda.cudaFree
import shape.komputation.layers.CudaEntryPointInstruction
import shape.komputation.layers.CudaForwardLayerInstruction
import shape.komputation.layers.Resourceful
import shape.komputation.loss.CudaLossFunctionInstruction
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

    fun forward(id : Int, input : Matrix) : Pointer {

        var output = this.entryPoint.forward(id, input)

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
        loss: CudaLossFunctionInstruction,
        numberIterations : Int,
        batchSize : Int,
        afterEachIteration : ((index : Int, loss : Double) -> Unit)? = null): Long {

        val lossFunction = loss.buildForCuda(this.cudaEnvironment)

        val numberExamples = inputs.size

        val batches = partitionIndices(numberExamples, batchSize)

        this.acquireLayerResources()

        if (lossFunction is Resourceful) {

            lossFunction.acquire()

        }

        val firstTarget = targets.first()
        val targetSize = firstTarget.numberRows * firstTarget.numberColumns

        val targetMemory = hashMapOf<Int, Pointer>()

        val start = System.currentTimeMillis()

        repeat(numberIterations) { indexIteration ->

            var id = 0

            var iterationLoss = 0.0

            for (batch in batches) {

                for (indexExample in batch) {

                    val input = inputs[indexExample]

                    val deviceTarget = if (targetMemory.containsKey(id)) {

                        targetMemory[id]!!

                    }
                    else {

                        val target = targets[indexExample]

                        val newPointer = Pointer()
                        copyFromHostToDevice(target.entries, targetSize, newPointer)

                        targetMemory[id] = newPointer

                        newPointer

                    }

                    val devicePointer = this.forward(id++, input)

                    lossFunction.accumulate(devicePointer, deviceTarget)

                    val lossGradient = lossFunction.backward(devicePointer, deviceTarget)

                    this.backward(lossGradient)

                }

                iterationLoss += lossFunction.accessAccumulation()

                lossFunction.reset()

                val scalingFactor = 1.0.div(batch.size.toDouble())

                this.optimize(scalingFactor)

            }

            if (afterEachIteration != null) {

                afterEachIteration(indexIteration, iterationLoss)

            }

        }

        val stop = System.currentTimeMillis()

        val time = stop - start

        targetMemory.values.forEach { pointer -> cudaFree(pointer) }

        if (lossFunction is Resourceful) {

            lossFunction.release()

        }

        this.releaseLayerResources()

        return time


    }

    fun optimize(scalingFactor : Double) {

        for (optimizable in this.optimizables) {

            optimizable.optimize(scalingFactor)

        }

    }

    fun acquireLayerResources() {

        if (this.entryPoint is Resourceful) {

            this.entryPoint.acquire()

        }

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

        if (this.entryPoint is Resourceful) {

            this.entryPoint.release()

        }

    }

}