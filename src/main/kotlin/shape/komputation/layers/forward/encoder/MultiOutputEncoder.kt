package shape.komputation.layers.forward.encoder

import shape.komputation.cpu.forward.encoder.CpuMultiOutputEncoder
import shape.komputation.cpu.forward.units.RecurrentUnit
import shape.komputation.layers.CpuForwardLayerInstruction

class MultiOutputEncoder(
    private val name: String?,
    private val unit: RecurrentUnit,
    private val numberSteps: Int,
    private val inputDimension: Int,
    private val hiddenDimension: Int,
    private val isReversed: Boolean) : CpuForwardLayerInstruction {

    override fun buildForCpu() =

        CpuMultiOutputEncoder(
            this.name,
            this.isReversed,
            this.unit,
            this.numberSteps,
            this.inputDimension,
            this.hiddenDimension
        )

}

fun multiOutputEncoder(
    unit : RecurrentUnit,
    numberSteps : Int,
    inputDimension : Int,
    hiddenDimension: Int,
    isReversed : Boolean = false) =

    multiOutputEncoder(
        null,
        unit,
        numberSteps,
        inputDimension,
        hiddenDimension,
        isReversed
    )

fun multiOutputEncoder(
    name : String?,
    unit : RecurrentUnit,
    numberSteps : Int,
    inputDimension : Int,
    hiddenDimension : Int,
    isReversed : Boolean = false) =

    MultiOutputEncoder(
        name,
        unit,
        numberSteps,
        inputDimension,
        hiddenDimension,
        isReversed
    )