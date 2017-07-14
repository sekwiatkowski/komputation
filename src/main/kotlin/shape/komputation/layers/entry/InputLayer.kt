package shape.komputation.layers.entry

import shape.komputation.cpu.layers.entry.CpuInputLayer
import shape.komputation.cuda.layers.entry.CudaInputLayer
import shape.komputation.layers.CpuEntryPointInstruction
import shape.komputation.layers.CudaEntryPointInstruction

class InputLayer(private val name : String? = null, private val dimension: Int) : CpuEntryPointInstruction, CudaEntryPointInstruction {

    override fun buildForCpu() =

        CpuInputLayer(this.name)

    override fun buildForCuda() =

        CudaInputLayer(this.dimension)

}

fun inputLayer(dimension : Int) = inputLayer(null, dimension)

fun inputLayer(name : String? = null, dimension : Int) = InputLayer(name, dimension)

