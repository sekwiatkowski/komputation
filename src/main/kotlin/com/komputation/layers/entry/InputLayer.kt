package com.komputation.layers.entry

import com.komputation.cpu.layers.entry.CpuInputLayer
import com.komputation.cuda.CudaContext
import com.komputation.cuda.layers.entry.CudaInputLayer
import com.komputation.layers.CpuEntryPointInstruction
import com.komputation.layers.CudaEntryPointInstruction

class InputLayer(private val name : String? = null, private val numberRows: Int, private val maximumLength: Int) : CpuEntryPointInstruction, CudaEntryPointInstruction {

    override fun buildForCpu() =
        CpuInputLayer(this.name)

    override fun buildForCuda(context: CudaContext) =
        CudaInputLayer(this.name, this.numberRows, this.maximumLength)

}

fun inputLayer(numberRows : Int, maximumLength: Int = 1) =
    inputLayer(null, numberRows, maximumLength)

fun inputLayer(name : String? = null, numberRows: Int, maximumLength: Int = 1) =
    InputLayer(name, numberRows, maximumLength)