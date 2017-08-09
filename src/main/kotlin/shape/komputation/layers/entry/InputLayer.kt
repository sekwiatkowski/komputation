package shape.komputation.layers.entry

import shape.komputation.cpu.layers.entry.CpuInputLayer
import shape.komputation.cuda.layers.entry.CudaInputLayer
import shape.komputation.layers.CpuEntryPointInstruction
import shape.komputation.layers.CudaEntryPointInstruction

class InputLayer(private val name : String? = null, private val numberRows: Int, private val numberColumns: Int) : CpuEntryPointInstruction, CudaEntryPointInstruction {

    override fun buildForCpu() =

        CpuInputLayer(this.name, this.numberRows, this.numberColumns)

    override fun buildForCuda() =

        CudaInputLayer(this.numberRows, this.numberColumns)

}

fun inputLayer(numberRows : Int, numberColumns : Int = 1) =

    inputLayer(null, numberRows, numberColumns)

fun inputLayer(name : String? = null, numberRows: Int, numberColumns: Int = 1) =

    InputLayer(name, numberRows, numberColumns)