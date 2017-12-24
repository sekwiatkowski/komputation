package com.komputation.cpu.layers.combination

import com.komputation.cpu.functions.add
import com.komputation.cpu.layers.CpuCombination
import com.komputation.cpu.layers.VariableLengthFloatArray
import com.komputation.cpu.layers.computePossibleLengths

/*
   Ex:
   f(x) = (g(x)+h(x))^2 = g(x)^2 + 2*g(x)*h(x) + h(x)^2
   d f(x) / g(x) = 2*g(x) + 2*h(x)
   d f(x) / h(x) = 2*h(x) + 2*g(x)
   chain = d (g(x)+h(x))^2 / d g(x)+h(x) = 2 * (g(x)+h(x)) = 2*g(x) + 2*h(x)
   chain * d g(x)+h(x) / d g(x) = chain
   chain * d g(x)+h(x) / d h(x) = chain
*/
class CpuAdditionCombination internal constructor(
    name : String? = null,
    private val numberRows : Int,
    private val minimumColumns : Int,
    private val maximumColumns : Int) : CpuCombination(name) {

    private val possibleLengths = computePossibleLengths(this.minimumColumns, this.maximumColumns)
    private val forwardResultStore = VariableLengthFloatArray(this.numberRows, this.possibleLengths)

    override fun forward(first: FloatArray, second: FloatArray, numberInputColumns : Int): FloatArray {
        val forwardResult = this.forwardResultStore.get(numberInputColumns)
        add(first, second, forwardResult, forwardResult.size)

        return forwardResult
    }

    // d (x + y) / d x = 1
    override fun backwardFirst(chain: FloatArray) =
        chain

    // d (x + y) / d y = 1
    override fun backwardSecond(chain: FloatArray) =
        chain
}