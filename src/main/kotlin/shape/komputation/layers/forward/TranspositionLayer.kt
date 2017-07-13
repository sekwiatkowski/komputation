package shape.komputation.layers.forward

import shape.komputation.cpu.forward.CpuTranspositionLayer
import shape.komputation.layers.CpuForwardLayerInstruction

class TranspositionLayer(private val name : String? = null) : CpuForwardLayerInstruction {

    override fun buildForCpu() =

        CpuTranspositionLayer(this.name)


}

fun transpositionLayer(name : String? = null) = TranspositionLayer(name)