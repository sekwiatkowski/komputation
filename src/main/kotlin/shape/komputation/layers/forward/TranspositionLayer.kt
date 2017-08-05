package shape.komputation.layers.forward

import shape.komputation.cpu.layers.forward.CpuTranspositionLayer
import shape.komputation.layers.CpuForwardLayerInstruction

class TranspositionLayer(private val name : String? = null, private val numberRows: Int, private val numberColumns: Int) : CpuForwardLayerInstruction {

    override fun buildForCpu() =

        CpuTranspositionLayer(this.name, this.numberRows, this.numberColumns)


}

fun transpositionLayer(numberRows : Int, numberColumns: Int) = TranspositionLayer(null, numberRows, numberColumns)

fun transpositionLayer(name : String? = null, numberRows : Int, numberColumns : Int) = TranspositionLayer(name, numberRows, numberColumns)