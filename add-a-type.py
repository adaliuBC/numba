from numba import types
from numba.extending import typeof_impl
from numba.extending import as_numba_type
from numba.extending import type_callable
from numba.extending import models, register_model
from numba.extending import make_attribute_wrapper
from numba.extending import overload_attribute
from numba.extending import lower_builtin
from numba.core import cgutils
from numba.extending import box, unbox, NativeValue

# def the class
class Interval(object):
    """
    A half-open interval on the real number line.
    """
    def __init__(self, lo, hi):
        self.lo = lo
        self.hi = hi

    def __repr__(self):
        return 'Interval(%f, %f)' % (self.lo, self.hi)

    @property
    def width(self):
        return self.hi - self.lo

# create a Numba type for the class
class IntervalType(types.Type):
    def __init__(self):
        super(IntervalType, self).__init__(name='Interval')

# use an obj instead of the type it self to represent the type
interval_type = IntervalType()

# telling numba any instance of `Interval` should be handled as interval_type
@typeof_impl.register(Interval)
def typeof_index(val, c):
    return interval_type

# telling numba any type of `Interval` should be handled as interval_type
as_numba_type.register(Interval, interval_type)

# ???
@type_callable(Interval)
def type_interval(context):
    def typer(lo, hi):
        if isinstance(lo, types.Float) and isinstance(hi, types.Float):
            return interval_type
    return typer

# def a local model for the class
@register_model(IntervalType)
class IntervalModel(models.StructModel):
    def __init__(self, dmm, fe_type):
        members = [
            ('lo', types.float64),
            ('hi', types.float64),
            ]
        models.StructModel.__init__(self, dmm, fe_type, members)

# make the attrs exposed
make_attribute_wrapper(IntervalType, 'lo', 'lo')
make_attribute_wrapper(IntervalType, 'hi', 'hi')

# equal to the @property width, exposing it as another attr
@overload_attribute(IntervalType, "width")
def get_width(interval):
    def getter(interval):
        return interval.hi - interval.lo
    return getter

# reimplement the constructor
@lower_builtin(Interval, types.Float, types.Float)
def impl_interval(context, builder, sig, args):
    typ = sig.return_type
    lo, hi = args
    # using create_struct_proxy to include more value types # ???
    interval = cgutils.create_struct_proxy(typ)(context, builder)
    interval.lo = lo
    interval.hi = hi
    return interval._getvalue()

@unbox(IntervalType)
def unbox_interval(typ, obj, c):
    """
    Convert a Interval object to a native interval structure.
    """
    # c.pyapi: changing IR to C obj?
    lo_obj = c.pyapi.object_getattr_string(obj, "lo")
    hi_obj = c.pyapi.object_getattr_string(obj, "hi")
    interval = cgutils.create_struct_proxy(typ)(c.context, c.builder)
    interval.lo = c.pyapi.float_as_double(lo_obj)
    interval.hi = c.pyapi.float_as_double(hi_obj)
    c.pyapi.decref(lo_obj)
    c.pyapi.decref(hi_obj)
    is_error = cgutils.is_not_null(c.builder, c.pyapi.err_occurred())
    return NativeValue(interval._getvalue(), is_error=is_error)

@box(IntervalType)
def box_interval(typ, val, c):
    """
    Convert a native interval structure to an Interval object.
    """
    interval = cgutils.create_struct_proxy(typ)(c.context, c.builder, value=val)
    lo_obj = c.pyapi.float_from_double(interval.lo)
    hi_obj = c.pyapi.float_from_double(interval.hi)
    class_obj = c.pyapi.unserialize(c.pyapi.serialize_object(Interval))
    res = c.pyapi.call_function_objargs(class_obj, (lo_obj, hi_obj))
    c.pyapi.decref(lo_obj)
    c.pyapi.decref(hi_obj)
    c.pyapi.decref(class_obj)
    return res


# usage
from numba import jit

@jit(nopython=True)
def inside_interval(interval, x):
    return interval.lo <= x < interval.hi

@jit(nopython=True)
def interval_width(interval):
    return interval.width

@jit(nopython=True)
def sum_intervals(i, j):
    return Interval(i.lo + j.lo, i.hi + j.hi)

int1 = Interval(1.0, 2.0)
int2 = Interval(2.0, 3.0)
int3 = sum_intervals(int1, int2)
print(int3.lo, int3.hi)