package shape.komputation.cpu.layers.forward.dropout

import shape.komputation.cpu.functions.*
import shape.komputation.cpu.layers.BaseCpuForwardLayer
import shape.komputation.layers.Resourceful
import shape.komputation.matrix.FloatMatrix
import java.util.*

class CpuDropoutLayer internal constructor(
    name: String?,
    private val numberEntries: Int,
    private val random: Random,
    private val keepProbability: Float) : BaseCpuForwardLayer(name), Resourceful {

    private var entrySeeds = IntArray(0)

    private var mask = BooleanArray(this.numberEntries)
    private var dropoutEntries = FloatArray(this.numberEntries)
    private var expectationEntries = FloatArray(this.numberEntries)
    private var backwardEntries = FloatArray(this.numberEntries)

    private val dropoutProbability = 1.0 - keepProbability

    private val threshold : Int

    init {

        val numberIntegers = Math.abs(Int.MIN_VALUE.toFloat()) + Int.MAX_VALUE.toFloat()

        val numberDropoutIntegers = (this.dropoutProbability * numberIntegers).toInt()
        this.threshold = Int.MIN_VALUE + numberDropoutIntegers

    }

    override fun acquire(maximumBatchSize: Int) {

        this.entrySeeds = IntArray(maximumBatchSize * this.numberEntries)

        seed(this.random, this.entrySeeds, this.numberEntries)

    }

    override fun release() {

        this.entrySeeds = IntArray(0)

    }

    override fun forward(withinBatch : Int, input: FloatMatrix, isTraining: Boolean) =

        if (isTraining) {

            val offset = withinBatch * this.numberEntries

            nextInteger(this.entrySeeds, offset, this.numberEntries)

            mask(this.entrySeeds, this.threshold, this.mask, offset, this.numberEntries)

            dropout(input.entries, this.mask, this.dropoutEntries, this.numberEntries)

            FloatMatrix(input.numberRows, input.numberColumns, this.dropoutEntries)

        }
        else {

            scale(input.entries, this.keepProbability, this.expectationEntries, this.numberEntries)

            FloatMatrix(input.numberRows, input.numberColumns, this.expectationEntries)

        }

    override fun backward(withinBatch : Int, chain: FloatMatrix): FloatMatrix {

        backwardDropout(chain.entries, this.mask, this.backwardEntries, this.numberEntries)

        return FloatMatrix(chain.numberRows, chain.numberColumns, this.backwardEntries)

    }

}