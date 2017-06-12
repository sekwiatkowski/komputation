package shape.komputation.optimization

typealias UpdateRule = (indexRow : Int, indexColumn : Int, current : Double, derivative : Double) -> Double