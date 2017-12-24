package com.komputation.instructions.continuation.projection

import com.komputation.instructions.ContinuationInstruction
import com.komputation.optimization.OptimizationInstruction

abstract class BaseWeighting(
    protected val name : String?,
    final override val numberOutputRows : Int,
    protected val optimizationStrategy : OptimizationInstruction? = null) : ContinuationInstruction {

    protected var numberInputRows = -1
    protected var minimumNumberInputColumns = -1
    protected var maximumNumberInputColumns = -1

    override fun setInputDimensionsFromPreviousInstruction(numberInputRows: Int, minimumNumberInputColumns: Int, maximumNumberInputColumns: Int) {
        this.numberInputRows = numberInputRows
        this.minimumNumberInputColumns = minimumNumberInputColumns
        this.maximumNumberInputColumns = maximumNumberInputColumns
    }
    override val minimumNumberOutputColumns
        get() = this.minimumNumberInputColumns
    override val maximumNumberOutputColumns
        get() = this.maximumNumberInputColumns

    protected val maximumNumberInputEntries
        get() = this.numberInputRows * this.maximumNumberInputColumns

    protected val numberWeightRows = this.numberOutputRows
    protected val numberWeightColumns
        get() = this.numberInputRows

}
