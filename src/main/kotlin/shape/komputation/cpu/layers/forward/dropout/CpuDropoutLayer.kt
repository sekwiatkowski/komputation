package shape.komputation.cpu.layers.forward.dropout

import shape.komputation.cpu.functions.backwardDropout
import shape.komputation.cpu.functions.dropout
import shape.komputation.cpu.functions.mask
import shape.komputation.cpu.functions.scale
import shape.komputation.cpu.layers.BaseCpuForwardLayer
import shape.komputation.matrix.FloatMatrix
import java.util.*

class CpuDropoutLayer internal constructor(
    name : String?,
    private val random : Random,
    private val numberEntries : Int,
    private val keepProbability : Float) : BaseCpuForwardLayer(name) {

    private var mask = BooleanArray(this.numberEntries)
    private var dropoutEntries = FloatArray(this.numberEntries)
    private var expectationEntries = FloatArray(this.numberEntries)
    private var backwardEntries = FloatArray(this.numberEntries)

    override fun forward(input: FloatMatrix, isTraining: Boolean) =

        if (isTraining) {

            mask(this.random, this.keepProbability, this.mask, this.numberEntries)

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