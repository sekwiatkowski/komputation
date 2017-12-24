package com.komputation.instructions.continuation

import com.komputation.instructions.ContinuationInstruction

abstract class BaseHigherOrderInstruction : ContinuationInstruction {

    protected abstract fun getLayers() : Array<out ContinuationInstruction>

    private val firstLayer
        get() = this.getLayers().first()
    private val lastLayer
        get() = this.getLayers().last()

    override fun setInputDimensionsFromPreviousInstruction(numberInputRows: Int, minimumNumberInputColumns: Int, maximumNumberInputColumns: Int) {
        this.firstLayer.setInputDimensionsFromPreviousInstruction(numberInputRows, minimumNumberInputColumns, maximumNumberInputColumns)

        val layers = this.getLayers()

        (1 until layers.size).forEach { index ->
            val first = layers[index-1]
            val second = layers[index]

            second.setInputDimensionsFromPreviousInstruction(first.numberOutputRows, first.minimumNumberOutputColumns, first.maximumNumberOutputColumns)
        }
    }

    override val numberOutputRows
        get() = this.lastLayer.numberOutputRows
    override val minimumNumberOutputColumns
        get() = this.lastLayer.minimumNumberOutputColumns
    override val maximumNumberOutputColumns
        get() = this.lastLayer.maximumNumberOutputColumns

}