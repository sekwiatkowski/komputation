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

fun negate(vector: DoubleArray) =

    DoubleArray(vector.size) { index ->

        -vector[index]

    }

fun scale(vector: DoubleArray, scalar : Double) =

    DoubleArray(vector.size) { index ->

        scalar * vector[index]

    }