package shape.konvolution.layers

import no.uib.cipr.matrix.Matrix
import shape.konvolution.BackwardResult
import shape.konvolution.Network
import shape.konvolution.concatColumns

class Branch(private val dimension: Int, vararg branches : Array<Layer>) : Layer {

    val numberBranches = branches.size

    val subnetworks = branches.map { layers -> Network(layers) }

    override fun forward(input: Matrix) =

        Array(numberBranches) { indexBranch ->

            subnetworks[indexBranch].forward(input).first()


        }
        .let { results ->

            concatColumns(results, numberBranches, 1, dimension)

        }

    override fun backward(input: Matrix, output : Matrix, chain : Matrix): BackwardResult {

        TODO("not implemented")
    }

}