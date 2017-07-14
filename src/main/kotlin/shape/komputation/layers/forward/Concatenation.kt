package shape.komputation.layers.forward

import shape.komputation.cpu.layers.forward.CpuConcatenation
import shape.komputation.layers.CpuForwardLayerInstruction

class Concatenation(private val name : String?, private val inputDimension : Int, private val continuations: Array<Array<CpuForwardLayerInstruction>>) : CpuForwardLayerInstruction {

    override fun buildForCpu() =

        CpuConcatenation(this.name, inputDimension, this.continuations)

}

fun concatenation(inputDimension : Int, vararg continuations: CpuForwardLayerInstruction) =

    concatenation(null, inputDimension, *continuations)

fun concatenation(name : String?, inputDimension : Int, vararg continuations: CpuForwardLayerInstruction) =

    concatenation(name, inputDimension, *continuations.map { layer -> arrayOf(layer) }.toTypedArray())

fun concatenation(inputDimension : Int, vararg continuations: Array<CpuForwardLayerInstruction>) =

    concatenation(null, inputDimension, *continuations)

fun concatenation(name : String?, inputDimension : Int, vararg continuations: Array<CpuForwardLayerInstruction>) =

    Concatenation(name, inputDimension, arrayOf(*continuations))