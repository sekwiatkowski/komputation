package com.komputation.instructions.entry

import com.komputation.cpu.instructions.CpuEntryPointInstruction
import com.komputation.cpu.layers.entry.CpuInput
import com.komputation.cuda.CudaContext
import com.komputation.cuda.instructions.CudaEntryPointInstruction
import com.komputation.cuda.layers.entry.CudaInput

class Input(
    private val name : String? = null,
    private val numberInputRows: Int,
    private val minimumInputColumns: Int,
    private val maximumInputColumns: Int) : CpuEntryPointInstruction, CudaEntryPointInstruction {

    override val numberOutputRows
        get() = this.numberInputRows
    override val minimumNumberOutputColumns
        get() = this.minimumInputColumns
    override val maximumNumberOutputColumns
        get() = this.maximumInputColumns

    override fun buildForCpu() =
        CpuInput(this.name)

    override fun buildForCuda(context: CudaContext) =
        CudaInput(this.name, this.numberInputRows, this.maximumInputColumns)

}

fun input(numberRows : Int, minimumColumns: Int = 1, maximumColumns: Int = minimumColumns) =
    input(null, numberRows, minimumColumns, maximumColumns)

fun input(name : String? = null, numberRows: Int, minimumColumns: Int = 1, maximumColumns: Int = minimumColumns) =
    Input(name, numberRows, minimumColumns, maximumColumns)