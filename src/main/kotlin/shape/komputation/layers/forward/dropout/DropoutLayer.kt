package shape.komputation.layers.forward.dropout

import shape.komputation.cpu.layers.forward.dropout.CpuDropoutLayer
import shape.komputation.layers.CpuForwardLayerInstruction
import java.util.*

class DropoutLayer(
    private val name : String?,
    private val random: Random,
    private val numberEntries: Int,
    private val keepProbability: Float) : CpuForwardLayerInstruction {

    override fun buildForCpu(): CpuDropoutLayer {

        return CpuDropoutLayer(this.name, this.random, this.numberEntries, this.keepProbability)

    }

}

fun dropoutLayer(random: Random, numberEntries: Int, keepProbability: Float) =

    dropoutLayer(null, random, numberEntries, keepProbability)

fun dropoutLayer(name: String?, random: Random, numberEntries: Int, keepProbability: Float) =

    DropoutLayer(name, random, numberEntries, keepProbability)