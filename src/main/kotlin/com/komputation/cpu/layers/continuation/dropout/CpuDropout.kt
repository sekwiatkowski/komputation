package com.komputation.cpu.layers.continuation.dropout

import com.komputation.cpu.functions.*
import com.komputation.cpu.layers.BaseCpuContinuationLayer
import com.komputation.instructions.Resourceful
import java.util.*

class CpuDropout internal constructor(
    name: String?,
    private val numberRows: Int,
    minimumColumns: Int,
    maximumColumns: Int,
    private val random: Random,
    private val keepProbability: Float) : BaseCpuContinuationLayer(name, numberRows, numberRows, minimumColumns, maximumColumns), Resourceful {

    private val threshold : Int
    private val dropoutProbability = 1.0 - keepProbability

    private val maximumEntries = this.numberRows * this.maximumColumns
    private var entrySeeds = IntArray(0)
    private var mask = BooleanArray(this.maximumEntries)

    init {
        val numberIntegers = Math.abs(Int.MIN_VALUE.toFloat()) + Int.MAX_VALUE.toFloat()

        val numberDropoutIntegers = (this.dropoutProbability * numberIntegers).toInt()
        this.threshold = Int.MIN_VALUE + numberDropoutIntegers
    }

    override fun acquire(maximumBatchSize: Int) {
        this.entrySeeds = IntArray(maximumBatchSize * this.maximumEntries)

        seed(this.random, this.entrySeeds, maximumBatchSize * this.maximumEntries)
    }

    override fun release() {
        this.entrySeeds = IntArray(0)
    }

    override fun computeForwardResult(withinBatch: Int, numberInputColumns: Int, input: FloatArray, forwardResult: FloatArray, isTraining: Boolean) {
        val numberEntries = this.numberRows * numberInputColumns

        if (isTraining) {
            val offset = withinBatch * this.maximumEntries

            nextInteger(this.entrySeeds, offset, numberEntries)

            mask(offset, numberEntries, this.entrySeeds, this.threshold, this.mask)

            dropout(numberEntries, input, this.mask, forwardResult)
        }
        else {
            scale(input, this.keepProbability, forwardResult, numberEntries)
        }
    }

    override fun computeBackwardResult(withinBatch: Int, numberInputColumns: Int, numberOutputColumns : Int, forwardResult: FloatArray, chain: FloatArray, backwardResult: FloatArray) {
        backwardDropout(chain, this.mask, backwardResult, backwardResult.size)
    }

}