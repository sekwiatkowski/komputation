package shape.komputation.layers.entry

import shape.komputation.matrix.IntegerMatrix
import shape.komputation.matrix.Matrix
import shape.komputation.matrix.RealMatrix
import shape.komputation.matrix.createRealMatrix
import shape.komputation.optimization.UpdateRule
import shape.komputation.optimization.updateSparsely

class LookupLayer(
    name : String?,
    private val data: Array<DoubleArray>,
    private val update: UpdateRule? = null) : EntryPoint(name), OptimizableEntryPoint {

    override fun forward(input : Matrix) : RealMatrix {

        input as IntegerMatrix

        val forwarded = Array(input.numberRows()) { index ->

            data[input.get(index, 0)]

        }

        return createRealMatrix(*forwarded)

    }

    override fun optimize(input : Matrix, chain: RealMatrix) {

        if (update != null) {

            input as IntegerMatrix

            val indices = input.getColumn(0)

            updateSparsely(data, indices, chain, update)

        }

    }

}

fun createLookupLayer(
    data: Array<DoubleArray>,
    optimizationStrategy : ((numberRows : Int, numberColumns : Int) -> UpdateRule)? = null): LookupLayer {

    return createLookupLayer(null, data, optimizationStrategy)
}

fun createLookupLayer(
    name : String? = null,
    data: Array<DoubleArray>,
    optimizationStrategy : ((numberRows : Int, numberColumns : Int) -> UpdateRule)? = null): LookupLayer {

    val updateRule = if (optimizationStrategy != null) {

        optimizationStrategy(data.size, data[0].size)

    }
    else {

        null
    }

    return LookupLayer(name, data, updateRule)

}