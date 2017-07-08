package shape.komputation.optimization.historical

import shape.komputation.optimization.UpdateRule

fun nesterov(learningRate: Double, momentum : Double): (Int, Int) -> UpdateRule {

    return { numberRows : Int, numberColumns : Int ->

        Nesterov(learningRate, momentum, numberRows * numberColumns)

    }

}

/*
    backup: v_backup = v
    history update: v = m * v - learning_rate * dx
    parameter update: x = x - m * v_backup + v + m * v
*/
class Nesterov(private val learningRate: Double, private val momentum: Double, size : Int) : UpdateRule {

    private val history = DoubleArray(size)
    private val backup = DoubleArray(size)

    override fun updateSparsely(start : Int, parameters: DoubleArray, gradient: DoubleArray, gradientSize : Int) {

        for(localIndex in 0..gradientSize-1) {

            val historyIndex = start + localIndex

            this.backup[historyIndex] = this.history[historyIndex]

            // Discount the previous history
            val updatedHistoryEntry = this.computeHistoryUpdate(historyIndex, gradient, localIndex)

            // Update the history
            this.history[historyIndex] = updatedHistoryEntry

            // Remove the look-ahead component from the parameter
            val removedPreviousLookAhead = parameters[localIndex] - this.momentum * this.backup[historyIndex]

            // Update the parameter and put it into the look-ahead position
            // updatedParameter = removedPreviousLookAhead + (1 + this.momentum) * updatedHistoryEntry
            //                  = removedPreviousLookAhead + updatedHistoryEntry + this.momentum * updatedHistory
            //                  = removedPreviousLookAhead + updatedHistoryEntry + newLookAhead
            parameters[localIndex] = removedPreviousLookAhead + (1 + this.momentum) * updatedHistoryEntry

        }

    }

    // Subtract the scaled gradient (looking ahead)
    private fun computeHistoryUpdate(historyIndex: Int, gradient: DoubleArray, localIndex: Int) =

        this.momentum * this.history[historyIndex] - this.learningRate * gradient[localIndex]

}