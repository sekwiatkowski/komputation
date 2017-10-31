package com.komputation.cpu.workflow

interface CpuClassificationTester {

    fun test(predictions : FloatArray, targets : FloatArray) : Boolean

}