package com.komputation.cuda.kernels

import org.apache.commons.io.IOUtils

class KernelFactory(private val capabilities : Pair<Int, Int>) {

    private fun read(relativePath: String) =

        IOUtils.toString(this.javaClass.getResourceAsStream("/cuda/$relativePath"), "UTF-8")

    fun create(instruction: KernelInstruction): Kernel {

        val sourceCode = read(instruction.relativePath)

        val relativeHeaderPaths = instruction.relativeHeaderPaths
        val includeNames = Array(relativeHeaderPaths.size) { index ->

            relativeHeaderPaths[index]

        }

        val headerFiles = Array(relativeHeaderPaths.size) { index -> read(relativeHeaderPaths[index]) }

        return Kernel(this.capabilities, sourceCode, instruction.name, instruction.nameExpression, headerFiles, includeNames)

    }

}