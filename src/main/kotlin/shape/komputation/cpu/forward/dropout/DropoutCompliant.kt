package shape.komputation.cpu.forward.dropout

import shape.komputation.cpu.SparseForwarding
import shape.komputation.cpu.forward.activation.CpuActivationLayer

interface DropoutCompliant : CpuActivationLayer, SparseForwarding