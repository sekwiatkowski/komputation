package shape.komputation.layers.forward

import shape.komputation.cpu.forward.CpuColumnRepetitionLayer
import shape.komputation.layers.CpuForwardLayerInstruction

class ColumnRepetitionLayer(private val name : String?, private val n : Int) : CpuForwardLayerInstruction {

    override fun buildForCpu() =

        CpuColumnRepetitionLayer(this.name, this.n)

}

fun columnRepetitionLayer(n : Int) =

    ColumnRepetitionLayer(null, n)

fun columnRepetitionLayer(name : String? = null, n : Int) =

    ColumnRepetitionLayer(name, n)