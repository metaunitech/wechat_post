# 探索AGI系列 | 03-01. 认知架构ACT-R理论与源码阅读 - Basic Utilities & Declarative Memory

---

## 系列前情提要

- 探索AGI系列 | 01. LLM不等于通用人工智能望周知
- 探索AGI系列 | 02. 智能体方法论：Agent智能体 & 认知架构（Cognitive Architecture）

## 前言

读者朋友们好，上一篇《探索AGI系列 | 02. 智能体方法论：Agent智能体 & 认知架构（Cognitive
Architecture》在各个平台都受到了比较大的反响，一方面体现出大家对于本系列的兴趣，另一方面也给了笔者更大的创作动力。因此笔者在忙完工作之余马不停蹄的为大家更新接下来的内容。再上一章末尾，笔者为大家介绍了一下认知模型ACT-R的基础构成，在这一章，我们将更深入的对ACT-R进行庖丁解牛，从理论和代码实现角度剖析ACT-R的思路。理论方面，笔者借鉴了论文：An Integrated Theory of the Mind. 笔者看到知乎平台上有中文翻译版本，有需要的朋友可以看这篇。
代码方面笔者阅读的是其python实现--PyACTR。原版实现应当是基于LISP的实现，Python版本与原版略有区别，但不影响我们的目的。

## Recap：ACT-R总览

![img_2.png](src/03/actr.png)

ACT-R（Adaptive Control of Thought—Rational）是一种认知架构，用于建模人类认知过程和行为。它旨在模拟人类思维和决策的过程，以便研究和理解认知心理学现象。ACT-R包括多个模块，每个模块都有不同的功能，用于模拟特定认知功能。

1. 意图模块 (Intentional Module / Goal Module)
   意图模块负责规划和控制行为。它确定当前任务的目标，协调其他模块的活动，以实现这些目标。意图模块是ACT-R中的执行控制中心。
2. 陈述性模块 (Declarative Module)
   陈述性模块用于存储事实、规则和概念的知识库。这允许模型访问和检索长期记忆中的信息，支持决策和问题解决过程。
3. 视觉模块 (Visual Module)
   视觉模块处理感觉输入，模拟人类的视觉处理过程。它负责感知和理解视觉信息，如物体、场景和符号。
4. 手动模块 (Manual Module)
   手动模块允许模型执行手部动作，例如移动、抓取物体等。它控制模型的运动和互动，模拟身体动作的执行。
5. 生产模块 (Production Module / Procedural System)
   生产模块是ACT-R的核心组成部分之一，用于表示知识和决策。它包括生产规则，描述了条件-动作对。当特定条件满足时，生产规则会触发执行相应的动作，模拟认知任务的决策过程。

ACT-R的相关开发人员定义其为Hybrid模式架构，其Production Module采用了Symbolic思想，简单来说就是一系列If-then语句，不涉及对大脑的模拟。Declarative
Module模块采用了Connectionist的思想，定义了Activation Score来作为记忆是否获取的标准。因而ACT-R整体应该是属于Hybrid。

## 模块详解

除了上述的五个模块，笔者在最开始要介绍三个"基础建设"单元：Buffer，Chunk和Productions。
在ACT-R认知架构中，Buffer和Chunk是两个关键概念，用于处理和存储信息，支持认知任务的执行和模拟。它们起到了重要的角色，有助于模型处理感知、记忆、决策和执行等认知功能。

### Chunk

Chunk是ACT-R中的信息单位，它是一个包含多个属性-值对的数据结构。每个属性-值对描述了信息的一部分，例如一个Chunk可以表示一个单词，它的属性可以包括"词汇"、"语义"、"频率"等等。Chunk用于表示和存储知识、事实、规则和任务状态。

在ACT-R中，Chunk可以存储在不同的Buffer中，以进行处理和传递。例如，视觉Buffer可以包含一个Chunk，表示一个模型看到的物体，听觉Buffer可以包含一个Chunk，表示听到的话语。Chunk的内容可以被提取、比较和操作，以支持认知任务的执行。

```Python
def chunktype(cls_name, field_names, defaults=None):
    """
    Creates type chunk. Works like namedtuple.

    For example:
    >>> chunktype('chunktype_example0', 'value')

    :param field_names: an iterable or a string of slot names separated by spaces
    :param defaults: default values for the slots, given as an iterable, counting from the last element
    """
    if cls_name in utilities.SPECIALCHUNKTYPES and field_names != utilities.SPECIALCHUNKTYPES[cls_name]:
        raise ACTRError("You cannot redefine slots of the chunk type '%s'; you can only use the slots '%s'" % (cls_name, utilities.SPECIALCHUNKTYPES[cls_name]))

    try:
        field_names = field_names.replace(',', ' ').split()
    except AttributeError:  # no .replace or .split
        pass  # assume it's already a sequence of identifiers
    field_names = tuple(sorted(name + "_" for name in field_names))
    for each in field_names:
        if each == "ISA" or each == "isa":
            raise ACTRError("You cannot use the slot 'isa' in your chunk. That slot is used to define chunktypes.")
    try:
        Chunk._chunktypes.update({cls_name:collections.namedtuple(cls_name, field_names, defaults=defaults)}) #chunktypes are not returned; they are stored as Chunk class attribute
    except TypeError:
        Chunk._chunktypes.update({cls_name:collections.namedtuple(cls_name, field_names)}) #chunktypes are not returned; they are stored as Chunk class attribute

class Chunk(Sequence):
    """
    ACT-R chunks. Based on namedtuple (tuple with dictionary-like properties).

    For example:
    >>> Chunk('chunktype_example0', value='one')
    chunktype_example0(value= one)
    """

    class EmptyValue(object):
        """
        Empty values used in chunks. These are None values.
        """
        pass
    
    _chunktypes = {}
    _undefinedchunktypecounter = 0
    _chunks = {}

    __emptyvalue = EmptyValue()

    _similarities = {} #dict of similarities between chunks

    def __init__(self, typename, **dictionary):
        self.typename = typename
        self.boundvars = {} #dict of bound variables

        kwargs = {}
        for key in dictionary:

            #change values (and values in a tuple) into string, when possible (when the value is not another chunk)
            if isinstance(dictionary[key], Chunk):
                dictionary[key] = utilities.VarvalClass(variables=None, values=dictionary[key], negvariables=(), negvalues=())

            elif isinstance(dictionary[key], utilities.VarvalClass):
                for x in dictionary[key]._fields:
                    if x in {"values", "variables"} and not isinstance(getattr(dictionary[key], x), str) and getattr(dictionary[key], x) != self.__emptyvalue and not isinstance(getattr(dictionary[key], x), Chunk):
                        raise TypeError("Values and variables must be strings, chunks or empty (None)")

                    elif x in {"negvariables", "negvalues"} and (not isinstance(getattr(dictionary[key], x), collections.abc.Sequence) or isinstance(getattr(dictionary[key], x), collections.abc.MutableSequence)):
                        raise TypeError("Negvalues and negvariables must be tuples")

            elif (isinstance(dictionary[key], collections.abc.Iterable) and not isinstance(dictionary[key], str)) or not isinstance(dictionary[key], collections.abc.Hashable):
                raise ValueError("The value of a chunk slot must be hashable and not iterable; you are using an illegal type for the value of the chunk slot %s, namely %s" % (key, type(dictionary[key])))

            else:
                #create namedtuple varval and split dictionary[key] into variables, values, negvariables, negvalues
                try:
                    temp_dict = utilities.stringsplitting(str(dictionary[key]))
                except utilities.ACTRError as e:
                    raise utilities.ACTRError("The chunk %s is not defined correctly; %s" %(dictionary[key], e))
                loop_dict = temp_dict.copy()
                for x in loop_dict:
                    if x == "negvariables" or x == "negvalues":
                        val = tuple(temp_dict[x])
                    else:
                        try:
                            val = temp_dict[x].pop()
                        except KeyError:
                            val = None
                    temp_dict[x] = val
                dictionary[key] = utilities.VarvalClass(**temp_dict)

            #adding _ to minimize/avoid name clashes
            kwargs[key+"_"] = dictionary[key]
        try:
            for elem in self._chunktypes[typename]._fields:

                if elem not in kwargs:

                    kwargs[elem] = self.__emptyvalue #emptyvalues are explicitly added to attributes that were left out
                    dictionary[elem[:-1]] = self.__emptyvalue #emptyvalues are also added to attributes in the original dictionary (since this might be used for chunktype creation later)

            if set(self._chunktypes[typename]._fields) != set(kwargs.keys()):

                chunktype(typename, dictionary.keys())  #If there are more args than in the original chunktype, chunktype has to be created again, with slots for new attributes
                warnings.warn("Chunk type %s is extended with new attributes" % typename)

        except KeyError:

            chunktype(typename, dictionary.keys())  #If chunktype completely missing, it is created first
            warnings.warn("Chunk type %s was not defined; added automatically" % typename)

        finally:
            self.actrchunk = self._chunktypes[typename](**kwargs)

        self.__empty = None #this will store what the chunk looks like without empty values (the values will be stored on the first call of the relevant function)
        self.__unused = None #this will store what the chunk looks like without unused values
        self.__hash = None, self.boundvars.copy() #this will store the hash along with variables (hash changes if some variables are resolved)
      
    def _asdict(self):
        """
        Create a dictionary out of chunk.
        """
        pass

    def __eq__(self, otherchunk):
        if hash(self) == hash(otherchunk):
            return True
        else:
            return False

    def __getattr__(self, name):
        if hasattr(self.actrchunk, name + "_"):
            return getattr(self.actrchunk, name + "_")
        else:
            raise AttributeError("Chunk has no such attribute")

    def __getitem__(self, pos):
        return re.sub("_$", "", self.actrchunk._fields[pos]), self.actrchunk[pos]

    def __hash__(self):
        pass

    def __iter__(self):
        for x, y in zip(self.actrchunk._fields, self.actrchunk):
            yield re.sub("_$", "", x), y

    def __len__(self):
        return len(self.actrchunk)

    def __repr__(self):
         pass

    def __lt__(self, otherchunk):
        """
        Check whether one chunk is proper part of another (given bound variables in boundvars).
        """
        return not self == otherchunk and self.match(otherchunk, partialmatching=False)

    def __le__(self, otherchunk):
        """
        Check whether one chunk is part of another (given boundvariables in boundvars).
        """
        return self == otherchunk or self.match(otherchunk, partialmatching=False) #actually, the second disjunct should be enough -- TODO: check why it fails in some cases; this might be important for partial matching

    def match(self, otherchunk, partialmatching, mismatch_penalty=1):
        """
        Check partial match (given bound variables in boundvars).
        """
        pass
      
    def removeempty(self):
        """
        Remove slot-value pairs that have the value __emptyvalue, that is, None and 'None'.
      
        Be careful! This returns a tuple with slot-value pairs.
        """
        pass

    def removeunused(self):
        """
        Remove values that were only added to fill in empty slots, using None. 
      
        Be careful! This returns a generator with slot-value pairs.
        """
        pass
```
上面附上Chunk实现的部分代码，其中笔者用pass忽略了具体的实现。从class的定义来看，Chunk类继承于Sequence类，并且在后面的代码中重载了一些魔法函数（一些软件工程上的优化例如防键值冲突等，与架构关系不大）。从Chunk的定义来看，chunk被定义为type和对应的slots，这点在代码中也有体现，Chunk类在__init__时，输入为typename和一些键值对。
```Python
class Chunk(Sequence):
    """
    ACT-R chunks. Based on namedtuple (tuple with dictionary-like properties).

    For example:
    >>> Chunk('chunktype_example0', value='one')
    chunktype_example0(value= one)
    """
    _chunktypes = {}
    _undefinedchunktypecounter = 0
    _chunks = {}

    __emptyvalue = EmptyValue()

    _similarities = {} #dict of similarities between chunks

    def __init__(self, typename, **dictionary):
        self.typename = typename
        self.boundvars = {} #dict of bound variables

        kwargs = {}
        for key in dictionary:
            #change values (and values in a tuple) into string, when possible (when the value is not another chunk)
            if isinstance(dictionary[key], Chunk):
                dictionary[key] = utilities.VarvalClass(variables=None, values=dictionary[key], negvariables=(), negvalues=())

            elif isinstance(dictionary[key], utilities.VarvalClass):
                for x in dictionary[key]._fields:
                    if x in {"values", "variables"} and not isinstance(getattr(dictionary[key], x), str) and getattr(dictionary[key], x) != self.__emptyvalue and not isinstance(getattr(dictionary[key], x), Chunk):
                        raise TypeError("Values and variables must be strings, chunks or empty (None)")

                    elif x in {"negvariables", "negvalues"} and (not isinstance(getattr(dictionary[key], x), collections.abc.Sequence) or isinstance(getattr(dictionary[key], x), collections.abc.MutableSequence)):
                        raise TypeError("Negvalues and negvariables must be tuples")

            elif (isinstance(dictionary[key], collections.abc.Iterable) and not isinstance(dictionary[key], str)) or not isinstance(dictionary[key], collections.abc.Hashable):
                raise ValueError("The value of a chunk slot must be hashable and not iterable; you are using an illegal type for the value of the chunk slot %s, namely %s" % (key, type(dictionary[key])))

            else:
                #create namedtuple varval and split dictionary[key] into variables, values, negvariables, negvalues
                try:
                    temp_dict = utilities.stringsplitting(str(dictionary[key]))
                except utilities.ACTRError as e:
                    raise utilities.ACTRError("The chunk %s is not defined correctly; %s" %(dictionary[key], e))
                loop_dict = temp_dict.copy()
                for x in loop_dict:
                    if x == "negvariables" or x == "negvalues":
                        val = tuple(temp_dict[x])
                    else:
                        try:
                            val = temp_dict[x].pop()
                        except KeyError:
                            val = None
                    temp_dict[x] = val
                dictionary[key] = utilities.VarvalClass(**temp_dict)

            #adding _ to minimize/avoid name clashes
            kwargs[key+"_"] = dictionary[key]
        try:
            for elem in self._chunktypes[typename]._fields:

                if elem not in kwargs:

                    kwargs[elem] = self.__emptyvalue #emptyvalues are explicitly added to attributes that were left out
                    dictionary[elem[:-1]] = self.__emptyvalue #emptyvalues are also added to attributes in the original dictionary (since this might be used for chunktype creation later)

            if set(self._chunktypes[typename]._fields) != set(kwargs.keys()):

                chunktype(typename, dictionary.keys())  #If there are more args than in the original chunktype, chunktype has to be created again, with slots for new attributes
                warnings.warn("Chunk type %s is extended with new attributes" % typename)

        except KeyError:

            chunktype(typename, dictionary.keys())  #If chunktype completely missing, it is created first
            warnings.warn("Chunk type %s was not defined; added automatically" % typename)

        finally:
            self.actrchunk = self._chunktypes[typename](**kwargs)

        self.__empty = None #this will store what the chunk looks like without empty values (the values will be stored on the first call of the relevant function)
        self.__unused = None #this will store what the chunk looks like without unused values
        self.__hash = None, self.boundvars.copy() #this will store the hash along with variables (hash changes if some variables are resolved)
      
```
### Buffer

Buffer是ACT-R中的信息处理单元，类似于短期工作记忆。每个Buffer都有特定的名称，表示不同的认知任务或处理阶段。以下是一些常见的Buffer：

视觉缓冲区 (Visual Buffer)：用于处理视觉输入，例如从眼睛接收到的图像信息。它允许模型暂时存储和处理视觉信息，以支持感知和理解。
听觉缓冲区 (Auditory Buffer)：用于处理听觉输入，例如听到的声音和语音。它允许模型存储和操作听觉信息，支持语言理解和音频处理。
手动缓冲区 (Manual Buffer)：用于处理手动操作，例如按下按钮或移动鼠标。它允许模型暂时存储手动操作的信息，以支持互动和执行动作。
陈述性记忆缓冲区 (Declarative Memory Buffer)：陈述性记忆缓冲区用于与长期记忆中的知识进行交互。它允许模型从长期记忆中检索和存储信息，支持记忆和知识的使用。

每个Buffer都可以包含一个或多个Chunk，这些Chunk是信息的基本单位。通过Buffer，ACT-R模型可以在不同的认知任务之间传递、存储和操作信息，从而模拟人类认知过程中的信息处理和转换。
