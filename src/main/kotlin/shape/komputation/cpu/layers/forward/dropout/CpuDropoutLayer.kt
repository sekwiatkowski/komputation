package shape.komputation.cpu.layers.forward.dropout

import shape.komputation.cpu.functions.generateMask
import shape.komputation.cpu.layers.BaseCpuForwardLayer
import shape.komputation.cpu.layers.combination.HadamardCombination
import shape.komputation.matrix.FloatMatrix
import shape.komputation.matrix.floatConstantColumnVector
import java.util.*

class CpuDropoutLayer internal constructor(
    name : String?,
    private val dimension : Int,
    private val random : Random,
    private val keepProbability : Float,
    private val activation: DropoutCompliant,
    private val takeExpectation : HadamardCombination) : BaseCpuForwardLayer(name) {

    private var mask = BooleanArray(0)
    private var expectation = floatConstantColumnVector(this.dimension, this.keepProbability)

    override fun forward(input: FloatMatrix, isTraining: Boolean) =

        if (isTraining) {

            this.mask = generateMask(this.dimension, this.random, this.keepProbability)

            // same as f(x) * m
            this.activation.forward(input, this.mask)

        }
        else {

            val activated = this.activation.forward(input, false)

            val tookExpectation = this.takeExpectation.forward(activated, this.expectation)

            tookExpectation

        }

    override fun backward(chain: FloatMatrix): FloatMatrix {

        // d f(x) * m / d f(x) = m

        val chainEntries = chain.entries

        val diffChainWrtSparseActivation = FloatMatrix(chain.numberRows, chain.numberColumns, FloatArray(this.dimension) { index ->

            if (this.mask[index]) {

                chainEntries[index]

            }
            else {

                0.0f

            }

        })

        return this.activation.backward(diffChainWrtSparseActivation)
    }

}