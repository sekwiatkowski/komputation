package shape.komputation.optimization

import shape.komputation.matrix.createRealMatrix

fun momentum(learningRate: Double, momentum: Double): (Int, Int) -> UpdateRule {

    return { numberRows : Int, numberColumns : Int ->

        val state = createRealMatrix(numberRows, numberColumns)

        val updateRule = { indexRow: Int, indexColumn: Int, current: Double, derivative: Double ->

            val newStateEntry = momentum * state.get(indexRow, indexColumn) + learningRate * derivative

            state.set(indexRow, indexColumn, newStateEntry)

            current - newStateEntry

        }

        updateRule


    }
}