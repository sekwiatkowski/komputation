package com.komputation.instructions

interface CanSetInputDimensions {
    fun setInputDimensionsFromPreviousInstruction(numberInputRows : Int, minimumNumberInputColumns : Int, maximumNumberInputColumns : Int)
}