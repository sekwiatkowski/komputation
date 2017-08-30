package shape.komputation.cuda.workflow

import jcuda.Pointer
import shape.komputation.cuda.CudaEvaluation
import shape.komputation.cuda.network.CudaForwardPropagator
import shape.komputation.cuda.memory.InputMemory
import shape.komputation.cuda.memory.TargetMemory
import shape.komputation.matrix.Matrix
import shape.komputation.matrix.partitionIndices

class CudaTester(
    private val forwardPropagator : CudaForwardPropagator,
    private val evaluation: CudaEvaluation,
    private val inputs : Array<Matrix>,
    private val targets: Array<FloatArray>,
    private val maximumBatchSize : Int) {

    private val numberInstances = inputs.size
    private val batches = partitionIndices(numberInstances, this.maximumBatchSize)

    private val inputMemory = InputMemory()
    private val targetMemory = TargetMemory(this.targets.first().size)

    init {

        this.evaluation.acquire(this.maximumBatchSize)

    }

    fun free() {

        this.inputMemory.free()
        this.targetMemory.free()

        this.evaluation.release()

    }

    fun run(): Float {

        this.evaluation.resetCount()

        for ((batchId, batch) in this.batches.withIndex()) {

            val currentBatchSize = batch.size

            val predictions = this.forwardPropagator.forward(batchId, currentBatchSize, batch, this.inputs, this.inputMemory,false)

            val pointerToPredictions = Pointer.to(predictions)

            val pointerToTargets = this.targetMemory.get(batchId, currentBatchSize, batch, this.targets)

            this.evaluation.evaluateBatch(currentBatchSize, pointerToPredictions, pointerToTargets)

        }

        val accuracy = this.evaluation.computeAccuracy()

        return accuracy

    }

}