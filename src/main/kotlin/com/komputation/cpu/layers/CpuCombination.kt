package com.komputation.cpu.layers

abstract class CpuCombination(private val name: String?) {

    abstract fun forward(first: FloatArray, second: FloatArray, numberInputColumns: Int) : FloatArray

    abstract fun backwardFirst(chain : FloatArray) : FloatArray

    abstract fun backwardSecond(chain : FloatArray) : FloatArray

}