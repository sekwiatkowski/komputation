package shape.komputation.matrix

fun intVector(vararg entries : Int) = IntMatrix(entries, entries.size, 1)

fun intScalar(scalar : Int) = intVector(scalar)