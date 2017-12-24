package com.komputation.cpu.optimization.historical

import com.komputation.cpu.optimization.UpdateRule

/*
    backup: v_backup = v
    history update: v = m * v - learning_rate * dx
    parameter update: x = x - m * v_backup + v + m * v
*/
class CpuNesterov(private val learningRate: Float, private val momentum: Float, size : Int) : UpdateRule {

    private val history = FloatArray(size)
    private val backup = FloatArray(size)

    override fun updateSparsely(start : Int, parameter: FloatArray, gradient: FloatArray, dimension: Int) {
        for(localIndex in 0 until dimension) {
            val historyIndex = start + localIndex

            val backup = this.history[historyIndex]

            this.backup[historyIndex] = backup

            // Discount the previous history
            val updatedHistoryEntry = this.computeHistoryUpdate(historyIndex, gradient[localIndex])

            // Update the history
            this.history[historyIndex] = updatedHistoryEntry

            // Remove the look-ahead component from the parameter
            val removedPreviousLookAhead = parameter[localIndex] - this.momentum * backup

            // Update the parameter and put it into the look-ahead position
            // updatedParameter = removedPreviousLookAhead + (1 + this.momentum) * updatedHistoryEntry
            //                  = removedPreviousLookAhead + updatedHistoryEntry + this.momentum * updatedHistory
            //                  = removedPreviousLookAhead + updatedHistoryEntry + newLookAhead
            parameter[localIndex] = removedPreviousLookAhead + (1.0f + this.momentum) * updatedHistoryEntry
        }
    }

    // Subtract the scaled gradient (looking ahead)
    private fun computeHistoryUpdate(historyIndex: Int, derivative: Float) =
        this.momentum * this.history[historyIndex] - this.learningRate * derivative

}