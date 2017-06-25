package shape.komputation.layers.combination

import shape.komputation.functions.hadamard
import shape.komputation.layers.CombinationLayer
import shape.komputation.matrix.DoubleMatrix

class HadamardCombination(val name: String?) : CombinationLayer(name) {

    private var first = DoubleArray(0)
    private var second = DoubleArray(0)
    private var numberRows = -1
    private var numberColumns = -1

    override fun forward(first: DoubleMatrix, second: DoubleMatrix): DoubleMatrix {

        this.first = first.entries
        this.second = second.entries

        this.numberRows = first.numberRows
        this.numberColumns = first.numberColumns

        return DoubleMatrix(this.numberRows, this.numberColumns, hadamard(first.entries, second.entries))

    }

    // d f(x) * g(x) / d f(x) = g(x)
    override fun backwardFirst(chain: DoubleMatrix) =

        DoubleMatrix(chain.numberRows, chain.numberColumns, hadamard(chain.entries, this.second))

    // d f(x) * g(x) / d g(x) = f(x)
    override fun backwardSecond(chain: DoubleMatrix) =

        DoubleMatrix(chain.numberRows, chain.numberColumns, hadamard(chain.entries, this.first))

}