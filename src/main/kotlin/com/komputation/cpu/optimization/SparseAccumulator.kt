package com.komputation.cpu.optimization

import com.komputation.matrix.constantIntArray
import java.util.*

class SparseAccumulator(numberVectors : Int, maximumBatchSize : Int, maximumLength : Int, private val dimension : Int) {

    private val hashTable = constantIntArray(numberVectors, -1)
    private val visited = BooleanArray(numberVectors)

    private val reverseHashTable = constantIntArray(maximumBatchSize * maximumLength, -1)
    private val counts = FloatArray(maximumBatchSize * maximumLength)
    private val sums = Array(maximumBatchSize * maximumLength) { FloatArray(dimension) }

    private var lastHashTableIndex = -1

    fun accumulate(parameterIndices: IntArray, numberParameterIndices: Int, gradient: FloatArray) {

        for (indexGradient in 0 until numberParameterIndices) {

            val parameterIndex = parameterIndices[indexGradient]

            val hashTableIndex = this.hash(parameterIndex)

            this.addToSum(indexGradient, hashTableIndex, gradient)

            // Is this the first occurrence of the vector?
            if (!this.visited[parameterIndex]) {
                // Increment the count
                this.counts[hashTableIndex] += 1.0f

                // Avoid further increments for the current example.
                this.visited[parameterIndex] = true
            }

        }

        // Reset the visit flag
        for (parameterIndex in parameterIndices) {
            this.visited[parameterIndex] = false
        }

    }

    private fun hash(parameterIndex: Int): Int {
        val existingHashTableIndex = this.hashTable[parameterIndex]

        if (existingHashTableIndex == -1) {
            val newHashTableIndex = ++this.lastHashTableIndex

            this.hashTable[parameterIndex] = newHashTableIndex
            this.reverseHashTable[newHashTableIndex] = parameterIndex

            return newHashTableIndex
        }
        else {
            return existingHashTableIndex
        }
    }

    private fun addToSum(indexGradient: Int, hashTableIndex: Int, gradient: FloatArray) {
        val sum = this.sums[hashTableIndex]

        val firstGradientEntryIndex = indexGradient * this.dimension

        for (indexDimension in 0 until this.dimension) {
            sum[indexDimension] += gradient[firstGradientEntryIndex + indexDimension]
        }
    }

    fun getSize() = this.lastHashTableIndex + 1

    fun getParameterIndices() = this.reverseHashTable

    fun getCounts() = this.counts

    fun getSums() = this.sums

    fun reset() {
        for (hashTableIndex in 0..this.lastHashTableIndex) {

            val parameterIndex = this.reverseHashTable[hashTableIndex]

            this.reverseHashTable[hashTableIndex] = -1
            this.hashTable[parameterIndex] = -1

            this.counts[hashTableIndex] = 0f

            Arrays.fill(this.sums[hashTableIndex], 0.0f)

        }

        this.lastHashTableIndex = -1
    }

}