package shape.komputation.layers.forward

import shape.komputation.cpu.layers.forward.CpuTranspositionLayer
import shape.komputation.layers.CpuForwardLayerInstruction

class TranspositionLayer(private val name : String? = null, private val numberEntries: Int) : CpuForwardLayerInstruction {

    override fun buildForCpu() =

        CpuTranspositionLayer(this.name, this.numberEntries)


}

fun transpositionLayer(numberEntries : Int) = TranspositionLayer(null, numberEntries)

fun transpositionLayer(name : String? = null, numberEntries : Int) = TranspositionLayer(name, numberEntries)