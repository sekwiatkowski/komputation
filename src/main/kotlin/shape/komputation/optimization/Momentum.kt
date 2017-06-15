package shape.komputation.optimization

import shape.komputation.matrix.createRealMatrix

fun momentum(learningRate: Double, momentum: Double): (Int, Int) -> UpdateRule {

    return { numberRows : Int, numberColumns : Int ->

        Momentum(learningRate, momentum, numberRows, numberColumns)

    }
}

class Momentum(private val learningRate: Double, private val momentum: Double, val numberRows : Int, val numberColumns : Int) : UpdateRule {

    val state = createRealMatrix(numberRows, numberColumns)

    override fun apply(index : Int, current: Double, derivative: Double): Double {

        val indexColumn = index % numberRows
        val indexRow = index / numberColumns

        val newStateEntry = momentum * state.get(indexRow, indexColumn) + learningRate * derivative

        state.set(indexRow, indexColumn, newStateEntry)

        return current - newStateEntry

    }

}