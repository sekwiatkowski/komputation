package shape.komputation.layers.forward.dropout

import shape.komputation.layers.Chainable
import shape.komputation.layers.DenseForwarding
import shape.komputation.layers.SparseForwarding

interface DropoutCompliant : DenseForwarding, SparseForwarding, Chainable