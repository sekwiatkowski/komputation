package shape.komputation.layers.forward

import shape.komputation.cpu.layers.forward.CpuCounterProbabilityLayer
import shape.komputation.layers.CpuForwardLayerInstruction

class CounterProbabilityLayer(
    private val name : String?,
    private val numberRows: Int,
    private val numberColumns: Int) : CpuForwardLayerInstruction {

    override fun buildForCpu() =

        CpuCounterProbabilityLayer(this.name, this.numberRows, this.numberColumns)

}


fun counterProbabilityLayer(numberRows: Int, numberColumns: Int) =

    counterProbabilityLayer(null, numberRows, numberColumns)

fun counterProbabilityLayer(name : String?, numberRows: Int, numberColumns: Int) =

    CounterProbabilityLayer(name, numberRows, numberColumns)