package shape.konvolution.layers.entry

import shape.konvolution.matrix.IntegerMatrix
import shape.konvolution.matrix.Matrix
import shape.konvolution.matrix.RealMatrix
import shape.konvolution.matrix.createRealMatrix
import shape.konvolution.optimization.UpdateRule
import shape.konvolution.optimization.updateSparsely

class LookupLayer(
    name : String?,
    private val data: Array<DoubleArray>,
    private val update: UpdateRule? = null) : EntryPoint(name), OptimizableEntryPoint {

    override fun forward() {

        val input = this.lastInput!!

        input as IntegerMatrix

        val forwarded = Array(input.numberRows()) { index ->

            data[input.get(index, 0)]

        }

        this.lastForwardResult = createRealMatrix(*forwarded)

    }

    override fun optimize(chain: RealMatrix) {

        if (update != null) {

            val input = this.lastInput as IntegerMatrix

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