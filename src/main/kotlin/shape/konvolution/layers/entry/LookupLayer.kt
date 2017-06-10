package shape.konvolution.layers.entry

import shape.konvolution.matrix.IntegerMatrix
import shape.konvolution.matrix.Matrix
import shape.konvolution.matrix.RealMatrix
import shape.konvolution.matrix.createRealMatrix
import shape.konvolution.optimization.Optimizer
import shape.konvolution.optimization.optimizeSparsely

class LookupLayer(
    private val data: Array<DoubleArray>,
    private val optimizer : Optimizer? = null) : EntryPoint, OptimizableEntryPoint {

    private val optimize = optimizer != null

    override fun forward(input : Matrix) : RealMatrix {

        input as IntegerMatrix

        val forwarded = Array(input.numberRows()) { index ->

            data[input.get(index, 0)]

        }

        return createRealMatrix(*forwarded)

    }

    override fun optimize(input: Matrix, output: Array<RealMatrix>, gradient: RealMatrix) {

        if (optimize) {

            val updatedEmbeddings = output.single().copy()

            this.optimizer!!.optimize(updatedEmbeddings, gradient)

            input as IntegerMatrix

            optimizeSparsely(data, input.getColumn(0), updatedEmbeddings)

        }

    }

}