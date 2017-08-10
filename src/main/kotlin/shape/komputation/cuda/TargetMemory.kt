package shape.komputation.cuda

import jcuda.Pointer
import jcuda.runtime.JCuda.cudaFree

class TargetMemory(private val targetSize : Int, private val targets : Array<FloatArray>, private val maximumBatchSize : Int) {

    private val memory = hashMapOf<Int, Pointer>()

    fun get(batchId : Int, batch : IntArray) =

        if (this.memory.containsKey(batchId)) {

            this.memory[batchId]!!

        }
        else {

            val batchTargetSize = this.maximumBatchSize * this.targetSize
            val batchTargets = FloatArray(batchTargetSize)

            for ((batchIndex, globalIndex) in batch.withIndex()) {

                val target = this.targets[globalIndex]

                System.arraycopy(target, 0, batchTargets, batchIndex * this.targetSize, this.targetSize)

            }

            val deviceTargets = Pointer()
            val result = setFloatArray(batchTargets, batchTargetSize, deviceTargets)

            val pointerToDeviceTargets = Pointer.to(deviceTargets)

            this.memory[batchId] = pointerToDeviceTargets

            pointerToDeviceTargets

        }

    fun release() {

        this.memory.values.forEach { pointer ->

            cudaFree(pointer)

        }

    }

}