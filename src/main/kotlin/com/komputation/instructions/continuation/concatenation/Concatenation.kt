package com.komputation.instructions.continuation.concatenation

import com.komputation.cpu.instructions.CpuContinuationInstruction
import com.komputation.cpu.layers.continuation.CpuConcatenation

class Concatenation internal constructor(
    private val name : String?,
    private val continuationInstructions: Array<out CpuContinuationInstruction>) : CpuContinuationInstruction {

    override fun setInputDimensionsFromPreviousInstruction(numberInputRows: Int, minimumNumberInputColumns: Int, maximumNumberInputColumns: Int) {
        this.continuationInstructions.forEach { layer ->
            layer.setInputDimensionsFromPreviousInstruction(numberInputRows, minimumNumberInputColumns, maximumNumberInputColumns)
        }
    }

    override val numberOutputRows
        get() = this.continuationInstructions.sumBy { it.numberOutputRows }
    override val minimumNumberOutputColumns
        get() = this.continuationInstructions[0].minimumNumberOutputColumns
    override val maximumNumberOutputColumns
        get() = this.continuationInstructions[0].maximumNumberOutputColumns

    override fun buildForCpu() =
        CpuConcatenation(this.name, this.continuationInstructions.map { instruction -> instruction.buildForCpu() }.toTypedArray())

}

fun concatenation(vararg continuations: CpuContinuationInstruction) =
    concatenation(null, *continuations)

fun concatenation(name : String?, vararg continuations: CpuContinuationInstruction) =
    Concatenation(name, continuations)