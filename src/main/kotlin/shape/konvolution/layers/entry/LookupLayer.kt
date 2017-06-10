package shape.konvolution.layers.entry

import shape.konvolution.matrix.IntegerMatrix
import shape.konvolution.matrix.Matrix
import shape.konvolution.matrix.RealMatrix
import shape.konvolution.matrix.createRealMatrix
import shape.konvolution.optimization.UpdateRule
import shape.konvolution.optimization.updateSparsely

class LookupLayer(
    private val data: Array<DoubleArray>,
    private val update: UpdateRule? = null) : EntryPoint, OptimizableEntryPoint {

    override fun forward(input : Matrix) : RealMatrix {

        input as IntegerMatrix

        val forwarded = Array(input.numberRows()) { index ->

            data[input.get(index, 0)]

        }

        return createRealMatrix(*forwarded)

    }

    override fun optimize(input: Matrix, output: RealMatrix, gradient: RealMatrix) {

        if (update != null) {

            input as IntegerMatrix

            val indices = input.getColumn(0)

            updateSparsely(data, indices, gradient, update)

        }

    }

}

fun createLookupLayer(
    data: Array<DoubleArray>,
    optimizationStrategy : ((numberRows : Int, numberColumns : Int) -> UpdateRule)? = null): LookupLayer {

    val updateRule = if (optimizationStrategy != null) {

        optimizationStrategy(data.size, data[0].size)

    }
    else {

        null
    }

    return LookupLayer(data, updateRule)

}