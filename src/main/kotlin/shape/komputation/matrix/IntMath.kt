package shape.komputation.matrix

object IntMath {

    fun ceil(x : Double) =

        Math.ceil(x).toInt()

    fun pow(base: Int, exponent: Int) =

        Math.pow(base.toDouble(), exponent.toDouble()).toInt()

    fun min(a: Int, b: Int) =

        if (b >= a) {

            a

        }
        else {

            b

        }


}