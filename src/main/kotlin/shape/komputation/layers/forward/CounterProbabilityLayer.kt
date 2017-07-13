package shape.komputation.layers.forward

import shape.komputation.cpu.layers.forward.CpuCounterProbabilityLayer
import shape.komputation.layers.CpuForwardLayerInstruction

class CounterProbabilityLayer(
    private val name : String?,
    private val dimension: Int) : CpuForwardLayerInstruction {

    override fun buildForCpu() =

        CpuCounterProbabilityLayer(this.name, this.dimension)

}


fun counterProbabilityLayer(dimension: Int) =

    counterProbabilityLayer(null, dimension)

fun counterProbabilityLayer(name : String?, dimension: Int) =

    CounterProbabilityLayer(name, dimension)