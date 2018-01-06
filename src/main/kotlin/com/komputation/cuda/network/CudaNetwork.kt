package com.komputation.cuda.network

import com.komputation.cuda.CudaContext
import com.komputation.cuda.instructions.CudaContinuationInstruction
import com.komputation.cuda.instructions.CudaEntryPointInstruction
import com.komputation.cuda.instructions.CudaLossFunctionInstruction
import com.komputation.cuda.kernels.TestingKernels
import com.komputation.cuda.layers.CudaContinuation
import com.komputation.cuda.layers.CudaEntryPoint
import com.komputation.cuda.memory.InputMemory
import com.komputation.cuda.setUpCudaContext
import com.komputation.cuda.workflow.CudaBinaryClassificationTester
import com.komputation.cuda.workflow.CudaMultiClassificationTester
import com.komputation.cuda.workflow.CudaTester
import com.komputation.cuda.workflow.CudaTrainer
import com.komputation.instructions.Resourceful
import com.komputation.instructions.acquireRecursively
import com.komputation.instructions.releaseRecursively
import com.komputation.matrix.Matrix
import com.komputation.optimization.Optimizable
import jcuda.Pointer
import jcuda.jcublas.JCublas2.cublasCreate
import jcuda.jcublas.JCublas2.cublasDestroy
import jcuda.jcublas.cublasHandle


class CudaNetwork internal constructor(
    private val maximumBatchSize: Int,
    entryPointInstruction: CudaEntryPointInstruction,
    vararg continuationInstructions: CudaContinuationInstruction) {

    private val cudaContext : CudaContext
    private val cublasHandle : cublasHandle

    private val entryPoint : CudaEntryPoint
    private val layers : Array<CudaContinuation>

    private val numberPredictionRows : Int
    private val minimumNumberPredictionColumns : Int
    private val maximumNumberPredictionColumns : Int

    init {
        this.cudaContext = setUpCudaContext()
        this.cublasHandle = cublasHandle()

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

        this.entryPoint = entryPointInstruction.buildForCuda(this.cudaContext)
        this.layers = Array(continuationInstructions.size) { index -> continuationInstructions[index].buildForCuda(this.cudaContext, this.cublasHandle) }
    }

    private val optimizables = listOf(this.entryPoint).plus(this.layers).filterIsInstance(Optimizable::class.java).reversed().toTypedArray()

    private val forwardPropagator =
        CudaForwardPropagator(this.entryPoint, this.layers)

    private val backwardPropagator =
        CudaBackwardPropagator(this.entryPoint, this.layers)

    init {
        cublasCreate(this.cublasHandle)

        acquireRecursively(this.entryPoint, this.maximumBatchSize)

        for (layer in this.layers) {
            acquireRecursively(layer, this.maximumBatchSize)
        }
    }

    fun getEntryPoint() =
        this.entryPoint

    fun getLayer(index : Int) =
        this.layers[index]

    fun free() {
        cublasDestroy(this.cublasHandle)

        for (layer in this.layers) {
            releaseRecursively(layer)
        }

        if (this.entryPoint is Resourceful) {
            this.entryPoint.release()
        }
    }

    fun training(
        inputs: Array<out Matrix>,
        targets: Array<FloatArray>,
        numberIterations : Int,
        lossFunction : CudaLossFunctionInstruction,
        afterEachIteration : ((index : Int, loss : Float) -> Unit)? = null): CudaTrainer {

        lossFunction.setInputDimensionsFromPreviousInstruction(this.numberPredictionRows, this.minimumNumberPredictionColumns, this.maximumNumberPredictionColumns)

        return CudaTrainer(
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
    }

    fun test(
        inputs: Array<out Matrix>,
        targets: Array<FloatArray>,
        batchSize: Int,
        numberCategories : Int,
        length : Int = 1) : CudaTester {

        val classificationTester =
            if (numberCategories == 1) {
                CudaBinaryClassificationTester(inputs.size, numberCategories, { this.cudaContext.createKernel(TestingKernels.binary()) })
            }
            else {
                CudaMultiClassificationTester(inputs.size, numberCategories, length, { this.cudaContext.createKernel(TestingKernels.multiClass()) })
            }


        return CudaTester(
            this.forwardPropagator,
            classificationTester,
            inputs,
            targets,
            batchSize
        )
    }

    fun predict(input : Matrix) : Pointer {
        val inputMemory = InputMemory()
        val prediction = this.forwardPropagator.forward(0, 1, intArrayOf(0), arrayOf(input), inputMemory, false)
        inputMemory.free()

        return prediction
    }

    fun predict(input : Array<out Matrix>) : Pointer {
        val inputMemory = InputMemory()
        val batchSize = input.size
        val prediction = this.forwardPropagator.forward(0, batchSize, IntArray(batchSize) { index -> index }, input, inputMemory, false)
        inputMemory.free()

        return prediction
    }

}

fun cudaNetwork(
    maximumBatchSize: Int,
    entryPointInstruction: CudaEntryPointInstruction,
    vararg continuationInstructions: CudaContinuationInstruction) =
    CudaNetwork(maximumBatchSize, entryPointInstruction, *continuationInstructions)