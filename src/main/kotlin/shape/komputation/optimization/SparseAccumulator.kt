package shape.komputation.optimization

import java.util.*

class SparseAccumulator(maximumBatchSize : Int) {

    private var inputs = Array(maximumBatchSize) { intArrayOf() }
    private var gradients = Array(maximumBatchSize) { doubleArrayOf() }

    private var count = 0

    fun accumulate(input: IntArray, gradient: DoubleArray) {

        this.inputs[count] = input
        this.gradients[count] = gradient

        this.count++

    }

    fun getAccumulation() : Pair<Array<IntArray>, Array<DoubleArray>> {

        val inputs = Arrays.copyOfRange(this.inputs, 0, this.count)
        val gradients = Arrays.copyOfRange(this.gradients, 0, this.count)

        return inputs to gradients

    }

    fun getCount() =

        this.count

    fun reset() {

        this.count = 0

    }

}