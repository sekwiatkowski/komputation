package shape.komputation.functions

fun selectEntries(input: DoubleArray, indices : IntArray) =

    DoubleArray(indices.size) { index -> input[indices[index]] }