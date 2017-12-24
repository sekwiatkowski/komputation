package com.komputation.instructions.continuation

import com.komputation.instructions.ContinuationInstruction

abstract class BaseEntrywiseInstruction : ContinuationInstruction {

    protected var numberInputRows = -1
    protected var minimumNumberInputColumns = -1
    protected var maximumNumberInputColumns = -1
    override fun setInputDimensionsFromPreviousInstruction(numberInputRows: Int, minimumNumberInputColumns: Int, maximumNumberInputColumns: Int) {
        this.numberInputRows = numberInputRows
        this.minimumNumberInputColumns = minimumNumberInputColumns
        this.maximumNumberInputColumns = maximumNumberInputColumns
    }

    override val numberOutputRows
        get() = this.numberInputRows
    override val minimumNumberOutputColumns
        get() = this.minimumNumberInputColumns
    override val maximumNumberOutputColumns
        get() = this.maximumNumberInputColumns

}