package shape.komputation.functions

fun add(a: DoubleArray, b: DoubleArray) =

    DoubleArray(a.size) { index ->

        a[index] + b[index]

    }

fun subtract(a: DoubleArray, b: DoubleArray) =

    DoubleArray(a.size) { index ->

        a[index] - b[index]

    }

fun hadamard(a: DoubleArray, b: DoubleArray) =

    DoubleArray(a.size) { index ->

        a[index] * b[index]

    }

fun negate(a: DoubleArray) =

    DoubleArray(a.size) { index ->

        -a[index]

    }