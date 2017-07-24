package shape.komputation.cpu.layers.combination

import shape.komputation.cpu.functions.hadamard
import shape.komputation.cpu.layers.CombinationLayer
import shape.komputation.matrix.FloatMatrix

class HadamardCombination internal constructor(name: String?, private val numberEntries : Int) : CombinationLayer(name) {

    private var first = FloatArray(0)
    private var second = FloatArray(0)
    private var forwardEntries = FloatArray(this.numberEntries)
    private var firstBackwardEntries = FloatArray(this.numberEntries)
    private var secondBackwardEntries = FloatArray(this.numberEntries)

    override fun forward(first: FloatMatrix, second: FloatMatrix): FloatMatrix {

        this.first = first.entries
        this.second = second.entries

        hadamard(first.entries, second.entries, this.forwardEntries, this.numberEntries)

        return FloatMatrix(first.numberRows, first.numberColumns, this.forwardEntries)

    }

    // d f(x) * g(x) / d f(x) = g(x)
    override fun backwardFirst(chain: FloatMatrix): FloatMatrix {

        hadamard(chain.entries, this.second, this.firstBackwardEntries, this.numberEntries)

        return FloatMatrix(chain.numberRows, chain.numberColumns, this.firstBackwardEntries)

    }

    // d f(x) * g(x) / d g(x) = f(x)
    override fun backwardSecond(chain: FloatMatrix): FloatMatrix {

        hadamard(chain.entries, this.first, this.secondBackwardEntries, this.numberEntries)

        return FloatMatrix(chain.numberRows, chain.numberColumns, this.secondBackwardEntries)

    }

}

fun hadamardCombination(numberEntries: Int) = hadamardCombination(null, numberEntries)

fun hadamardCombination(name : String? = null, numberEntries: Int) = HadamardCombination(name, numberEntries)