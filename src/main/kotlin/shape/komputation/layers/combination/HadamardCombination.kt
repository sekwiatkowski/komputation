package shape.komputation.layers.combination

import shape.komputation.functions.hadamard
import shape.komputation.layers.CombinationLayer
import shape.komputation.matrix.DoubleMatrix

// a * b
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

    override fun backwardFirst(chain: DoubleMatrix) =

        DoubleMatrix(chain.numberRows, chain.numberColumns, this.second)

    override fun backwardSecond(chain: DoubleMatrix) =

        DoubleMatrix(chain.numberRows, chain.numberColumns, this.first)

}