package shape.konvolution.layers.continuation

import shape.konvolution.BackwardResult
import shape.konvolution.Network
import shape.konvolution.matrix.RealMatrix
import shape.konvolution.matrix.concatColumns
import shape.konvolution.layers.entry.InputLayer

class Branch(private val dimension: Int, vararg branches : Array<ContinuationLayer>) : ContinuationLayer {

    val numberBranches = branches.size

    val subnetworks = branches.map { layers -> Network(InputLayer(), *layers) }

    override fun forward(input: RealMatrix): Array<RealMatrix> {

        val branches = Array(numberBranches) { indexBranch ->

            subnetworks[indexBranch].forward(input).first().last()

        }

        val concatenation = concatColumns(branches, numberBranches, 1, dimension)

        return arrayOf(concatenation)

    }

    override fun backward(inputs: Array<RealMatrix>, outputs : Array<RealMatrix>, chain : RealMatrix): BackwardResult {

        TODO("not implemented")
    }

}