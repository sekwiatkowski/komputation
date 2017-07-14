package shape.komputation.matrix

fun intColumnVector(vararg entries : Int) = IntMatrix(entries.size, 1, entries)

fun intScalar(scalar : Int) = intColumnVector(scalar)