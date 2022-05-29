package torch

import org.bytedeco.javacpp.*
import org.bytedeco.javacpp.indexer.Indexer
import org.bytedeco.pytorch
import org.bytedeco.pytorch.global.torch as torch_native
import org.bytedeco.pytorch.{LinearImpl, LogSoftmaxFuncOptions, Module}

import scala.annotation.{targetName, varargs}
import scala.reflect.ClassTag

def eye(n: Long): Tensor = Tensor(torch_native.eye(n))
def ones(size: Long*): Tensor = Tensor(torch_native.ones(size*))