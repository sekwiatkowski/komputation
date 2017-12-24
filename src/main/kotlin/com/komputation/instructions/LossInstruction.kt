package com.komputation.instructions

interface LossInstruction : CanSetInputDimensions

abstract class BaseLossInstruction(val name : String?) : LossInstruction {

    protected var numberInputRows = -1
    protected var minimumNumberInputColumns = -1
    protected var maximumNumberInputColumns = -1

    override fun setInputDimensionsFromPreviousInstruction(numberInputRows: Int, minimumNumberInputColumns: Int, maximumNumberInputColumns: Int) {
        this.numberInputRows = numberInputRows
        this.minimumNumberInputColumns = minimumNumberInputColumns
        this.maximumNumberInputColumns = maximumNumberInputColumns
    }

}