package shape.komputation.cpu.optimization

import java.util.*

class SparseAccumulator(numberVectors : Int, maximumBatchSize : Int, maximumLength : Int, private val dimension : Int) {

    private val idMapping = IntArray(numberVectors) { -1 }
    private val visited = BooleanArray(numberVectors)

    private val ids = IntArray(maximumBatchSize * maximumLength)
    private val counts = FloatArray(maximumBatchSize * maximumLength)
    private val sums = Array(maximumBatchSize * maximumLength) { FloatArray(dimension) }

    private var lastNewId = -1

    fun accumulate(ids: IntArray, lastIndex : Int, gradient: FloatArray) {

        var start = 0

        for (indexId in 0..lastIndex) {

            val id = ids[indexId]

            var mappedId = this.idMapping[id]

            if (mappedId == -1) {

                mappedId = ++lastNewId

                this.idMapping[id] = mappedId
                this.ids[mappedId] = id

            }

            val sum = this.sums[mappedId]

            for (indexDimension in 0..this.dimension - 1) {

                sum[indexDimension] += gradient[start++]

            }

            // Is this the first occurrence of the vector?
            if (!this.visited[id]) {

                // Increment the count
                this.counts[mappedId] += 1.0f

                // Avoid further increments for the current example.
                this.visited[id] = true

            }

        }

        // Reset the visit flag
        for (id in ids) {

            this.visited[id] = false

        }

    }

    fun getSize() = this.lastNewId + 1

    fun getIds() = this.ids

    fun getCounts() = this.counts

    fun getSums() = this.sums

    fun reset() {

        for (id in this.ids) {

            this.idMapping[id] = -1

        }

        for (index in 0..lastNewId) {

            this.counts[index] = 0.0f

            Arrays.fill(this.sums[index], 0.0f)

        }

        this.lastNewId = -1

    }

}