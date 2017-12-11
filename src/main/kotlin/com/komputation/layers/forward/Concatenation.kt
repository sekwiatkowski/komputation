package com.komputation.layers.forward

import com.komputation.cpu.layers.forward.CpuConcatenation
import com.komputation.layers.CpuForwardLayerInstruction

class Concatenation internal constructor(
    private val name : String?,
    private val layerInstructions: Array<out CpuForwardLayerInstruction>) : CpuForwardLayerInstruction {

    override fun buildForCpu() =
        CpuConcatenation(this.name, this.layerInstructions.map { instruction -> instruction.buildForCpu() }.toTypedArray())

}

fun concatenation(vararg layers: CpuForwardLayerInstruction) =
    concatenation(null, *layers)

fun concatenation(name : String?, vararg layers: CpuForwardLayerInstruction) =
    Concatenation(name, layers)