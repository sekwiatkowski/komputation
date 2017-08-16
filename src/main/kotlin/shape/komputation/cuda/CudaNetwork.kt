package shape.komputation.cuda

import jcuda.Pointer
import jcuda.jcublas.JCublas2.cublasCreate
import jcuda.jcublas.JCublas2.cublasDestroy
import jcuda.jcublas.cublasHandle
import shape.komputation.cuda.kernels.EvaluationKernels
import shape.komputation.cuda.workflow.CudaTester
import shape.komputation.cuda.workflow.CudaTrainer
import shape.komputation.layers.CudaEntryPointInstruction
import shape.komputation.layers.CudaForwardLayerInstruction
import shape.komputation.layers.Resourceful
import shape.komputation.loss.CudaLossFunctionInstruction
import shape.komputation.matrix.Matrix
import shape.komputation.optimization.Optimizable

class CudaNetwork(
    private val maximumBatchSize: Int,
    entryPointInstruction: CudaEntryPointInstruction,
    vararg forwardLayerInstructions: CudaForwardLayerInstruction) {

    private val cudaContext = setUpCudaContext()
    private val cublasHandle = cublasHandle()

    private val entryPoint = entryPointInstruction.buildForCuda()

    private val layers = forwardLayerInstructions.map { it.buildForCuda(this.cudaContext, this.cublasHandle) }
    private val numberLayers = this.layers.size
    private val optimizables = listOf(this.entryPoint).plus(this.layers).filterIsInstance(Optimizable::class.java).reversed()

    fun forward(batchId: Int, batchSize: Int, indices: IntArray, inputs: Array<Matrix>, inputMemory : InputMemory, isTraining: Boolean) : Pointer {

        var output = this.entryPoint.forward(batchId, batchSize, indices, inputs, inputMemory)

        for (layer in this.layers) {

            output = layer.forward(output, batchSize, isTraining)

        }

        return output

    }

    fun backward(lossGradient : Pointer, batchSize: Int): Pointer {

        var chain = lossGradient

        for(indexLayer in this.numberLayers - 1 downTo 0) {

            val layer = this.layers[indexLayer]

            chain = layer.backward(chain, batchSize)

        }

        val floatArray = getFloatArray(chain, 4 * 784)

        return this.entryPoint.backward(chain)

    }

    init {

        cublasCreate(this.cublasHandle)

        if (this.entryPoint is Resourceful) {

            this.entryPoint.acquire(maximumBatchSize)

        }

        for (layer in this.layers) {

            if (layer is Resourceful) {

                layer.acquire(maximumBatchSize)

            }

        }

    }

    fun free() {

        cublasDestroy(this.cublasHandle)

        for (layer in this.layers) {

            if (layer is Resourceful) {

                layer.release()

            }

        }

        if (this.entryPoint is Resourceful) {

            this.entryPoint.release()

        }

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
        lossFunction : CudaLossFunctionInstruction,
        afterEachIteration : ((index : Int, loss : Float) -> Unit)? = null) =

        CudaTrainer(
            this,
            inputs,
            targets,
            numberIterations,
            this.maximumBatchSize,
            lossFunction.buildForCuda(this.cudaContext),
            afterEachIteration
        )


    fun test(
        inputs: Array<Matrix>,
        targets: Array<FloatArray>,
        batchSize: Int,
        numberCategories : Int,
        length : Int = 1) =

        CudaTester(
            this,
            CudaEvaluation(inputs.size, numberCategories, length, { this.cudaContext.createKernel(EvaluationKernels.evaluation()) }),
            inputs,
            targets,
            batchSize
        )

}