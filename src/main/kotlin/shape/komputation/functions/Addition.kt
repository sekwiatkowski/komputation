package shape.komputation.functions

fun add(a: DoubleArray, b: DoubleArray) =

    DoubleArray(a.size) { index ->

        a[index] + b[index]

    }