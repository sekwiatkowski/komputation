package shape.komputation.layers.forward

import shape.komputation.cpu.forward.CpuConcatenation
import shape.komputation.layers.CpuForwardLayerInstruction

class Concatenation(private val name : String?, private val continuations: Array<Array<CpuForwardLayerInstruction>>) : CpuForwardLayerInstruction {

    override fun buildForCpu() =

        CpuConcatenation(this.name, this.continuations)

}

fun concatenation(vararg continuations: CpuForwardLayerInstruction) =

    concatenation(null, *continuations)

fun concatenation(name : String?, vararg continuations: CpuForwardLayerInstruction) =

    concatenation(name, *continuations.map { layer -> arrayOf(layer) }.toTypedArray())

fun concatenation(vararg continuations: Array<CpuForwardLayerInstruction>) =

    concatenation(null, *continuations)

fun concatenation(name : String?, vararg continuations: Array<CpuForwardLayerInstruction>) =

    Concatenation(name, arrayOf(*continuations))