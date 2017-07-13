package shape.komputation.layers.forward.dropout

import shape.komputation.cpu.combination.hadamardCombination
import shape.komputation.cpu.forward.dropout.CpuDropoutLayer
import shape.komputation.layers.CpuDropoutCompliantInstruction
import shape.komputation.layers.CpuForwardLayerInstruction
import shape.komputation.layers.concatenateNames
import java.util.*

class DropoutLayer(
    private val name : String?,
    private val dimension: Int,
    private val random: Random,
    private val keepProbability: Double,
    private val activation: CpuDropoutCompliantInstruction) : CpuForwardLayerInstruction {

    override fun buildForCpu(): CpuDropoutLayer {

        val takeExpectationName = concatenateNames(this.name, "take-expectation")
        val takeExpectation = hadamardCombination(takeExpectationName)

        val activation = this.activation.buildForCpu()

        return CpuDropoutLayer(this.name, this.dimension, this.random, this.keepProbability, activation, takeExpectation)

    }

}

fun dropoutLayer(dimension: Int, random: Random, keepProbability: Double, activation: CpuDropoutCompliantInstruction) =

    dropoutLayer(null, dimension, random, keepProbability, activation)

fun dropoutLayer(name : String?, dimension: Int, random: Random, keepProbability: Double, activation: CpuDropoutCompliantInstruction) =

    DropoutLayer(name, dimension, random, keepProbability, activation)