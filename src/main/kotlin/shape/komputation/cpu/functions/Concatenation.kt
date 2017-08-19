package shape.komputation.cpu.functions

fun sparselyPadAndConcatenateIntMatrixEntries(matrices : Array<IntArray>, maximumLength: Int, result : IntArray) {

    for ((withinBatch, array) in matrices.withIndex()) {

        val paddedInputEntries = IntArray(maximumLength)
        pad(array, array.size, maximumLength, -1, paddedInputEntries)

        System.arraycopy(paddedInputEntries, 0, result, withinBatch * maximumLength, maximumLength)

    }

}

fun pad(entries : IntArray, currentLength : Int, maximumLength: Int, symbol : Int, result : IntArray) {

    for (index in 0 until currentLength) {

        result[index] = entries[index]

    }

    for (index in currentLength until maximumLength) {

        result[index] = symbol

    }

}

fun sparselyConcatenateFloatMatrixEntries(matrices : Array<FloatArray>, maximumNumberEntries: Int, result : FloatArray) {

    for ((withinBatch, array) in matrices.withIndex()) {

        System.arraycopy(array, 0, result, withinBatch * maximumNumberEntries, array.size)

    }

}

fun denselyConcatenateFloatArrays(arrays : Array<FloatArray>, numberEntriesPerArray : Int, result : FloatArray) {

    for ((withinBatch, array) in arrays.withIndex()) {

        System.arraycopy(array, 0, result, withinBatch * numberEntriesPerArray, numberEntriesPerArray)

    }

}