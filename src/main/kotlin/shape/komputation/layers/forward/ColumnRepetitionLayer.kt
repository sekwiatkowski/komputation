package shape.komputation.layers.forward

import shape.komputation.cpu.layers.forward.CpuColumnRepetitionLayer
import shape.komputation.layers.CpuForwardLayerInstruction

class ColumnRepetitionLayer(private val name : String?, private val numberRows : Int, private val numberColumns : Int) : CpuForwardLayerInstruction {

    override fun buildForCpu() =

        CpuColumnRepetitionLayer(this.name, this.numberRows, this.numberColumns)

}

fun columnRepetitionLayer(numberRows : Int, numberColumns : Int) =

    ColumnRepetitionLayer(null, numberRows, numberColumns)

fun columnRepetitionLayer(name : String? = null, numberRows : Int, numberColumns : Int) =

    ColumnRepetitionLayer(name, numberRows, numberColumns)