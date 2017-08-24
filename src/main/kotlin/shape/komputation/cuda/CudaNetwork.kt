package shape.komputation.cuda

import jcuda.Pointer
import jcuda.jcublas.JCublas2.cublasCreate
import jcuda.jcublas.JCublas2.cublasDestroy
import jcuda.jcublas.cublasHandle
import shape.komputation.cuda.kernels.EvaluationKernels
import shape.komputation.cuda.layers.CudaEntryPoint
import shape.komputation.cuda.layers.CudaForwardLayer
import shape.komputation.cuda.memory.InputMemory
import shape.komputation.cuda.workflow.CudaTester
import shape.komputation.cuda.workflow.CudaTrainer
import shape.komputation.layers.CudaEntryPointInstruction
import shape.komputation.layers.CudaForwardLayerInstruction
import shape.komputation.layers.Resourceful
import shape.komputation.layers.acquireRecursively
import shape.komputation.loss.CudaLossFunctionInstruction
import shape.komputation.matrix.Matrix
import shape.komputation.optimization.Optimizable

class CudaForwardPropagator(
    private val entryPoint: CudaEntryPoint,
    private val layers : Array<CudaForwardLayer>) {

    fun forward(batchId: Int, batchSize: Int, indices: IntArray, inputs: Array<Matrix>, memory : InputMemory, isTraining: Boolean) : Pointer {

        var previousLayerState = this.entryPoint.forward(batchId, batchSize, indices, inputs, memory)

        for (layer in this.layers) {

            previousLayerState = layer.forward(batchSize, previousLayerState, isTraining)

        }

        return previousLayerState

    }

}

class CudaBackwardPropagator(
    private val entryPoint: CudaEntryPoint,
    private val layers : Array<CudaForwardLayer>) {

    private val numberLayers = this.layers.size

    fun backward(lossGradient : Pointer, batchSize: Int): Pointer {

        var chain = lossGradient

        for(indexLayer in this.numberLayers - 1 downTo 0) {

            val layer = this.layers[indexLayer]

            chain = layer.backward(batchSize, chain)

        }

        return this.entryPoint.backward(chain)

    }

}

class CudaNetwork(
    private val maximumBatchSize: Int,
    entryPointInstruction: CudaEntryPointInstruction,
    vararg forwardLayerInstructions: CudaForwardLayerInstruction) {

    private val cudaContext = setUpCudaContext()
    private val cublasHandle = cublasHandle()

    private val entryPoint = entryPointInstruction.buildForCuda(this.cudaContext)
    private val layers = Array(forwardLayerInstructions.size) { index -> forwardLayerInstructions[index].buildForCuda(this.cudaContext, this.cublasHandle) }
    private val optimizables = listOf(this.entryPoint).plus(this.layers).filterIsInstance(Optimizable::class.java).reversed().toTypedArray()

    private val forwardPropagator = CudaForwardPropagator(this.entryPoint, this.layers)
    private val backwardPropagator = CudaBackwardPropagator(this.entryPoint, this.layers)

    init {

        cublasCreate(this.cublasHandle)

        acquireRecursively(this.entryPoint, this.maximumBatchSize)

        for (layer in this.layers) {

            acquireRecursively(layer, this.maximumBatchSize)

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

    fun training(
        inputs: Array<Matrix>,
        targets: Array<FloatArray>,
        numberIterations : Int,
        lossFunction : CudaLossFunctionInstruction,
        afterEachIteration : ((index : Int, loss : Float) -> Unit)? = null) =

        CudaTrainer(
            this.forwardPropagator,
            this.backwardPropagator,
            this.optimizables,
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
            this.forwardPropagator,
            CudaEvaluation(inputs.size, numberCategories, length, { this.cudaContext.createKernel(EvaluationKernels.evaluation()) }),
            inputs,
            targets,
            batchSize
        )

}