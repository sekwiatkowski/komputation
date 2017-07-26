package shape.komputation.cpu.layers.forward.dropout

import shape.komputation.cpu.functions.*
import shape.komputation.cpu.layers.BaseCpuForwardLayer
import shape.komputation.matrix.FloatMatrix
import java.util.*

class CpuDropoutLayer internal constructor(
    name: String?,
    private val numberEntries: Int,
    random: Random,
    private val keepProbability: Float) : BaseCpuForwardLayer(name) {

    private val mask = BooleanArray(this.numberEntries)
    private val dropoutEntries = FloatArray(this.numberEntries)
    private val expectationEntries = FloatArray(this.numberEntries)
    private val backwardEntries = FloatArray(this.numberEntries)

    private val entrySeeds = IntArray(this.numberEntries)
    private val dropoutProbability = 1.0 - keepProbability

    private val threshold : Int

    init {

        val numberIntegers = Math.abs(Int.MIN_VALUE.toFloat()) + Int.MAX_VALUE.toFloat()
        val numberDropoutIntegers = (this.dropoutProbability * numberIntegers).toInt()
        this.threshold = Int.MIN_VALUE + numberDropoutIntegers

        seed(random, this.entrySeeds, this.numberEntries)

    }

    override fun forward(input: FloatMatrix, isTraining: Boolean) =

        if (isTraining) {

            mask(this.entrySeeds, this.threshold, this.mask, this.numberEntries)

            dropout(input.entries, this.mask, this.dropoutEntries, this.numberEntries)

            FloatMatrix(input.numberRows, input.numberColumns, this.dropoutEntries)

        }
        else {

            scale(input.entries, this.keepProbability, this.expectationEntries, this.numberEntries)

            FloatMatrix(input.numberRows, input.numberColumns, this.expectationEntries)

        }

    override fun backward(chain: FloatMatrix): FloatMatrix {

        backwardDropout(chain.entries, this.mask, this.backwardEntries, this.numberEntries)

        return FloatMatrix(chain.numberRows, chain.numberColumns, this.backwardEntries)

    }

}