package shape.komputation.optimization

fun nesterov(learningRate: Double, momentum : Double): (Int, Int) -> UpdateRule {

    return { numberRows : Int, numberColumns : Int ->

        Nesterov(learningRate, momentum, numberRows, numberColumns)

    }

}

/*
    backup: v_backup = v
    history update: v = m * v - learning_rate * dx
    parameter update: x = x - m * v_backup + v + m * v
*/

class Nesterov(private val learningRate: Double, private val momentum: Double, val numberRows : Int, val numberColumns : Int) : UpdateRule {

    private val history = DoubleArray(numberRows * numberColumns)
    private val backup = DoubleArray(numberRows * numberColumns)

    override fun apply(index : Int, current: Double, derivative: Double): Double {

        this.backup[index] = this.history[index]

        // Discount the previous history
        // Subtract the scaled gradient (looking ahead)
        val updatedHistoryEntry = this.momentum * this.history[index] - this.learningRate * derivative

        // Update the history
        this.history[index] = updatedHistoryEntry

        // Remove the look-ahead component from the parameter
        val removedPreviousLookAhead = current - this.momentum * this.backup[index]

        // Update the parameter and put it into the look-ahead position
        // updatedParameter = removedPreviousLookAhead + (1 + this.momentum) * updatedHistoryEntry
        //                  = removedPreviousLookAhead + updatedHistoryEntry + this.momentum * updatedHistory
        //                  = removedPreviousLookAhead + updatedHistoryEntry + newLookAhead
        val updatedParameter = removedPreviousLookAhead + (1 + this.momentum) * updatedHistoryEntry

        return updatedParameter
    }

}