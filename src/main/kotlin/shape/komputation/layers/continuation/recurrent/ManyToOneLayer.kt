package shape.komputation.layers.continuation.recurrent

import shape.komputation.layers.continuation.ContinuationLayer
import shape.komputation.layers.continuation.OptimizableContinuationLayer
import shape.komputation.matrix.RealMatrix

class ManyToOneLayer(
    name : String?,
    private val cell: RecurrentCell) : ContinuationLayer(name), OptimizableContinuationLayer {

    // activate(stateful_projection(input))
    override fun forward(input : RealMatrix) : RealMatrix {

        for (indexColumn in 0..input.numberColumns() - 2) {

            val column = input.getColumn(indexColumn)

            cell.forward(column)

        }

        return cell.forward(input.getColumn(input.numberColumns()-1))
    }

    override fun backward(chain: RealMatrix) : RealMatrix {

        TODO("")

    }

    override fun optimize() {

        TODO("")

    }

}