package shape.komputation.layers.forward.activation

import shape.komputation.cpu.forward.activation.CpuReluLayer
import shape.komputation.layers.CpuDropoutCompliantInstruction

class ReluLayer(private val name : String?) : CpuDropoutCompliantInstruction {

    override fun buildForCpu() =

        CpuReluLayer(this.name)

}

fun reluLayer(name : String? = null) = ReluLayer(name)