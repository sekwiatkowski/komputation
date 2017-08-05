package shape.komputation.cpu.layers.combination

import shape.komputation.cpu.functions.hadamard
import shape.komputation.cpu.layers.CombinationLayer
import shape.komputation.matrix.FloatMatrix

class HadamardCombination internal constructor(
    name: String?,
    private val numberRows : Int,
    private val numberColumns : Int) : CombinationLayer(name) {

    private val numberEntries = this.numberRows * this.numberColumns

    private var firstEntries = FloatArray(0)
    private var secondEntries = FloatArray(0)

    private var forwardEntries = FloatArray(this.numberEntries)
    private var firstBackwardEntries = FloatArray(this.numberEntries)
    private var secondBackwardEntries = FloatArray(this.numberEntries)

    override fun forward(first: FloatMatrix, second: FloatMatrix): FloatMatrix {

        this.firstEntries = first.entries
        this.secondEntries = second.entries

        hadamard(first.entries, second.entries, this.forwardEntries, this.numberEntries)

        return FloatMatrix(this.numberRows, this.numberColumns, this.forwardEntries)

    }

    // d f(x) * g(x) / d f(x) = g(x)
    override fun backwardFirst(chain: FloatMatrix): FloatMatrix {

        hadamard(chain.entries, this.secondEntries, this.firstBackwardEntries, this.numberEntries)

        return FloatMatrix(chain.numberRows, chain.numberColumns, this.firstBackwardEntries)

    }

    // d f(x) * g(x) / d g(x) = f(x)
    override fun backwardSecond(chain: FloatMatrix): FloatMatrix {

        hadamard(chain.entries, this.firstEntries, this.secondBackwardEntries, this.numberEntries)

        return FloatMatrix(chain.numberRows, chain.numberColumns, this.secondBackwardEntries)

    }

}

fun hadamardCombination(numberRows: Int, numberColumns: Int) = hadamardCombination(null, numberRows, numberColumns)

fun hadamardCombination(name : String? = null, numberRows: Int, numberColumns: Int) = HadamardCombination(name, numberRows, numberColumns)