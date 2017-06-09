package shape.konvolution.layers.continuation

import shape.konvolution.BackwardResult
import shape.konvolution.Network
import shape.konvolution.RealMatrix
import shape.konvolution.concatColumns
import shape.konvolution.layers.entry.InputLayer

class Branch(private val dimension: Int, vararg branches : Array<ContinuationLayer>) : ContinuationLayer {

    val numberBranches = branches.size

    val subnetworks = branches.map { layers -> Network(InputLayer(), *layers) }

    override fun forward(input: RealMatrix) =

        Array(numberBranches) { indexBranch ->

            subnetworks[indexBranch].forward(input).first()


        }
        .let { results ->

            concatColumns(results, numberBranches, 1, dimension)

        }

    override fun backward(input: RealMatrix, output : RealMatrix, chain : RealMatrix): BackwardResult {

        TODO("not implemented")
    }

}