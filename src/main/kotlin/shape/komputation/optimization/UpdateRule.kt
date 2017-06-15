package shape.komputation.optimization

interface UpdateRule {

    fun apply (index : Int, current : Double, derivative : Double): Double

}