package shape.komputation.layers.forward

import shape.komputation.cpu.layers.forward.CpuConcatenation
import shape.komputation.layers.CpuForwardLayerInstruction

class Concatenation(private val name : String?, private val numberRows : Int, private val numberColumns : Int, private val continuations: Array<Array<CpuForwardLayerInstruction>>) : CpuForwardLayerInstruction {

    override fun buildForCpu() =

        CpuConcatenation(this.name, this.numberRows, this.numberColumns, this.continuations)

}

fun concatenation(numberRows: Int, numberColumns: Int, vararg continuations: CpuForwardLayerInstruction) =

    concatenation(null, numberRows, numberColumns, *continuations)

fun concatenation(name : String?, numberRows: Int, numberColumns: Int, vararg continuations: CpuForwardLayerInstruction) =

    concatenation(name, numberRows, numberColumns, *continuations.map { layer -> arrayOf(layer) }.toTypedArray())

fun concatenation(numberRows: Int, numberColumns: Int, vararg continuations: Array<CpuForwardLayerInstruction>) =

    concatenation(null, numberRows, numberColumns, *continuations)

fun concatenation(name : String?, numberRows: Int, numberColumns: Int, vararg continuations: Array<CpuForwardLayerInstruction>) =

    Concatenation(name, numberRows, numberColumns, arrayOf(*continuations))