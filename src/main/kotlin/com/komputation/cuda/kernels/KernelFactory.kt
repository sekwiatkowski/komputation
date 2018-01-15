package com.komputation.cuda.kernels

import org.apache.commons.io.IOUtils

class KernelFactory(private val capabilities : Pair<Int, Int>) {

    private fun changeToParentDirectory(path : String, numberChanges : Int)  =
        path
            .split("/")
            .dropLast(numberChanges)
            .joinToString("/")

    private val includeRegex = "#include \"(?<header>.*)\"".toRegex()

    private fun countSubstrings(string: String, substring: String) =
        string.split(substring).size - 1

    private fun read(relativePath: String)  =
        IOUtils.toString(this.javaClass.getResourceAsStream("/cuda/$relativePath"), "UTF-8")

    private fun removeIncludesAndCollectHeaders(
        isHeader : Boolean,
        relativePathToBase : String,
        source: String,
        headers : HashMap<String, String>): String {

        val lines = source.split(System.lineSeparator())

        val relativePathToBaseWithoutFilename = changeToParentDirectory(relativePathToBase, 1)

        val processedLines = lines
            .map { line ->

                val matchResult = includeRegex.matchEntire(line)

                if (matchResult != null) {

                    val includeDirective = matchResult.groups["header"]!!.value

                    val numberOfChangesToParentDirectory = countSubstrings(includeDirective, "../")
                    val converted = if (numberOfChangesToParentDirectory == 0) {
                        relativePathToBaseWithoutFilename
                    }
                    else {
                        val changedRelativePath = changeToParentDirectory(relativePathToBaseWithoutFilename, numberOfChangesToParentDirectory)
                        changedRelativePath
                    }

                    val rest = includeDirective.substringAfterLast("../")

                    val headerPathRelativeToBase = if (converted.isEmpty()) {
                        rest
                    }
                    else {
                        converted + "/" + rest
                    }

                    val headerSource = read(headerPathRelativeToBase)

                    removeIncludesAndCollectHeaders(true, headerPathRelativeToBase, headerSource, headers)

                    null
                }
                else {
                    line
                }

            }
            .filterNotNull()

        val processedCode = processedLines.joinToString(System.lineSeparator())

        if (isHeader) {
            headers[relativePathToBase] = processedCode
        }

        return processedCode

    }

    fun create(instruction: KernelInstruction): Kernel {
        val relativePath = instruction.relativePath

        val sourceCode = read(relativePath)

        val headers = linkedMapOf<String, String>()
        val sourceWithoutIncludes = removeIncludesAndCollectHeaders(false, relativePath, sourceCode, headers)

        headers.remove("cuda.h")

        val headerList = headers.toList()
        val numberOfHeaders = headers.size
        val headerIncludes = Array(numberOfHeaders) { index -> headerList[index].first }
        val headerSources = Array(numberOfHeaders) { index -> headerList[index].second }

        val includes = headerIncludes
            .joinToString(System.lineSeparator()) { includePathRelativeToBase ->
                "#include \"$includePathRelativeToBase\""
            }

        val finalSourceCode = includes + System.lineSeparator() + sourceWithoutIncludes

        return Kernel(this.capabilities, finalSourceCode, instruction.name, headerSources, headerIncludes)
    }

}