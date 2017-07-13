package shape.komputation.cpu.forward.dropout

import shape.komputation.cpu.BaseForwardLayer
import shape.komputation.cpu.combination.HadamardCombination
import shape.komputation.functions.generateMask
import shape.komputation.matrix.DoubleMatrix
import shape.komputation.matrix.doubleConstantColumnVector
import java.util.*

class CpuDropoutLayer internal constructor(
    name : String?,
    private val dimension : Int,
    private val random : Random,
    private val keepProbability : Double,
    private val activation: DropoutCompliant,
    private val takeExpectation : HadamardCombination) : BaseForwardLayer(name) {

    private var mask = BooleanArray(0)
    private var expectation = doubleConstantColumnVector(this.dimension, this.keepProbability)

    override fun forward(input: DoubleMatrix, isTraining: Boolean) =

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

    override fun backward(chain: DoubleMatrix): DoubleMatrix {

        // d f(x) * m / d f(x) = m

        val chainEntries = chain.entries

        val diffChainWrtSparseActivation = DoubleMatrix(chain.numberRows, chain.numberColumns, DoubleArray(this.dimension) { index ->

            if (this.mask[index]) {
                chainEntries[index]
            } else {
                0.0
            }

        })

        return this.activation.backward(diffChainWrtSparseActivation)
    }

}