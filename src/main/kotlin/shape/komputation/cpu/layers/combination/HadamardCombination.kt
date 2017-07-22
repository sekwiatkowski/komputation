package shape.komputation.cpu.layers.combination

import shape.komputation.cpu.functions.hadamard
import shape.komputation.cpu.layers.CombinationLayer
import shape.komputation.matrix.FloatMatrix

class HadamardCombination internal constructor(val name: String?) : CombinationLayer(name) {

    private var first = FloatArray(0)
    private var second = FloatArray(0)

    override fun forward(first: FloatMatrix, second: FloatMatrix): FloatMatrix {

        this.first = first.entries
        this.second = second.entries

        return FloatMatrix(first.numberRows, first.numberColumns, hadamard(first.entries, second.entries))

    }

    // d f(x) * g(x) / d f(x) = g(x)
    override fun backwardFirst(chain: FloatMatrix) =

        FloatMatrix(chain.numberRows, chain.numberColumns, hadamard(chain.entries, this.second))

    // d f(x) * g(x) / d g(x) = f(x)
    override fun backwardSecond(chain: FloatMatrix) =

        FloatMatrix(chain.numberRows, chain.numberColumns, hadamard(chain.entries, this.first))

}

fun hadamardCombination(name : String? = null) = HadamardCombination(name)