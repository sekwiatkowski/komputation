package shape.komputation.cpu.functions

fun add(a: FloatArray, b: FloatArray) =

    FloatArray(a.size) { index ->

        a[index] + b[index]

    }

fun subtract(a: FloatArray, b: FloatArray) =

    FloatArray(a.size) { index ->

        a[index] - b[index]

    }

fun hadamard(a: FloatArray, b: FloatArray) =

    FloatArray(a.size) { index ->

        a[index] * b[index]

    }

fun negate(vector: FloatArray) =

    FloatArray(vector.size) { index ->

        -vector[index]

    }

fun scale(vector: FloatArray, scalar : Float) =

    FloatArray(vector.size) { index ->

        scalar * vector[index]

    }