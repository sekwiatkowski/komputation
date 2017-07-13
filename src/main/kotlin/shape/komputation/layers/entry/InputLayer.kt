package shape.komputation.layers.entry

import shape.komputation.cpu.layers.entry.CpuInputLayer
import shape.komputation.layers.CpuEntryPointInstruction

class InputLayer(private val name : String? = null) : CpuEntryPointInstruction {

    override fun buildForCpu() =

        CpuInputLayer(this.name)


}

fun inputLayer(name : String? = null) = InputLayer(name)