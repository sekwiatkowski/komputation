package shape.komputation.matrix

object IntMath {

    fun ceil(x : Double) =

        Math.ceil(x).toInt()

    fun closestPowerOfTwo(x: Int) =

        Math.pow(2.0, Math.ceil(Math.log(x.toDouble()) / Math.log(2.0))).toInt()

    fun pow(base: Int, exponent: Int) =

        Math.pow(base.toDouble(), exponent.toDouble()).toInt()

    fun min(a: Int, b: Int) =

        if (a >= b) {

            b

        }
        else {

            a

        }

    fun max(a: Int, b: Int) =

        if (a >= b) {

            a

        }
        else {

            b

        }

}