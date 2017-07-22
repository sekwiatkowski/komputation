package shape.komputation.cpu.functions

fun selectEntries(input: FloatArray, indices : IntArray) =

    FloatArray(indices.size) { index -> input[indices[index]] }