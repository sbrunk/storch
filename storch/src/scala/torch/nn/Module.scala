package torch
package nn

import org.bytedeco.javacpp.CharPointer
import org.bytedeco.pytorch
import org.bytedeco.pytorch.{Conv2dImpl, InputArchive, OutputArchive}

import java.nio.CharBuffer
import scala.reflect.ClassTag
import scala.collection.immutable.TreeSeqMap
import scala.collection.immutable.SeqMap

abstract class Module extends Cloneable {

  protected[torch] var _nativeModule                   = pytorch.Module()
  private[torch] def nativeModule: pytorch.Module      = _nativeModule // = pytorch.Module()
  private var childModules: TreeSeqMap[String, Module] = TreeSeqMap.empty

  def namedBuffers(recurse: Boolean = true): SeqMap[String, Tensor[_]] =
    val buffers = nativeModule.named_buffers(recurse)
    TreeSeqMap.from((0 until buffers.size().toInt).map { i =>
      val item = buffers.get(i)
      (item.key().getString(), Tensor.apply[DType](item.access()))
    })

  def namedParameters(recurse: Boolean = true): SeqMap[String, Tensor[_]] =
    val params = nativeModule.named_parameters(recurse)
    TreeSeqMap.from((0 until params.size().toInt).map { i =>
      val item = params.get(i)
      (item.key().getString(), Tensor.apply[DType](item.access()))
    })

  def parameters: Seq[Tensor[_]] = parameters(recurse = true)

  def parameters(recurse: Boolean): Seq[Tensor[_]] =
    nativeModule.parameters().get().map(Tensor.apply[DType](_))

  // TODO make strict a parameter
  // TODO improve error handling
  def loadStateDict(stateDict: Map[String, Tensor[DType]]): Unit =
    val tensorsToLoad = namedParameters() ++ namedBuffers()
    //assert(stateDict.keySet -- tensorsToLoad.keySet == Set.empty, s"keys missing in state dict: ${tensorsToLoad.keySet -- stateDict.keySet}")
    for ((key, param) <- tensorsToLoad if stateDict.contains(key))
      noGrad {
        param.copy_(stateDict(key))
      }

  def modules(recurse: Boolean): Seq[Module] =
    childModules.values.flatMap(child => child +: child.modules).toSeq.distinct
  def modules: Seq[Module] = modules(recurse = true)

  def namedChildren: SeqMap[String, Module] = childModules
  def namedModules: SeqMap[String, Module]  = namedChildren.flatMap((name, module) => module.namedModules)

  def copy(): this.type =
    val clone = super.clone().asInstanceOf[Module]
    clone._nativeModule = _nativeModule.clone()
    clone.asInstanceOf[this.type]

  protected[torch] def registerWithParent[T <: pytorch.Module](parent: T)(using name: sourcecode.Name): Unit =
    parent.register_module(name.value, nativeModule)

  def register[M <: Module](child: M)(using name: sourcecode.Name) =
    // println(s"registering ${name.value}:$child")
    childModules = childModules.updated(name.value, child)
    child.registerWithParent(this.nativeModule)
    child

  def register[D <: DType](t: Tensor[D], requiresGrad: Boolean = true)(using name: sourcecode.Name): Tensor[D] =
    nativeModule.register_parameter(name.value, t.native, requiresGrad)
    t

  def eval(): Unit = nativeModule.eval()

  def isTraining: Boolean = nativeModule.is_training

  def train(on: Boolean = true): Unit = nativeModule.train(on)

  def to(device: Device): this.type =
    //val nativeCopy = nativeModule.clone()
    nativeModule.asModule.to(device.toNative)
    // copy
    // val clone: this.type = copy()
    // clone.nativeModule = nativeCopy
    this

  def to(dtype: DType, nonBlocking: Boolean = false): this.type =
    val nativeCopy = nativeModule.clone()
    nativeCopy.asModule.to(dtype.toScalarType)
    this

  def save(outputArchive: OutputArchive) = nativeModule.save(outputArchive)

  def load(inputArchive: InputArchive) = nativeModule.load(inputArchive)

  override def toString(): String = getClass().getSimpleName()

  private def doSummarize(indent: Int): String =
    val thisModule = toString
    if modules.isEmpty then thisModule
    else
      thisModule + namedChildren
        .map((name, module) => s"${" " * (indent + 2)}($name): " + module.doSummarize(indent + 2))
        .mkString("(\n", "\n", s"\n${" " * indent})")
  def summarize: String =
    doSummarize(0)
}

trait HasParams[ParamType <: FloatNN | ComplexNN] extends Module:
  override def parameters(recurse: Boolean): Seq[Tensor[ParamType]] =
    nativeModule.parameters(recurse).get().toSeq.map(Tensor.apply[ParamType])
  override def parameters: Seq[Tensor[ParamType]] = parameters(recurse = true)
  transparent inline def paramType = deriveDType[ParamType]

trait HasWeight[ParamType <: FloatNN]:
  def weight: Tensor[ParamType]

trait TensorModule[D <: DType] extends Module with (Tensor[D] => Tensor[D]):
  override def toString(): String = "TensorModule"
