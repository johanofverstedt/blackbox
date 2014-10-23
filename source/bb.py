
"""
- BlackBox (bb) -
My attempt to learn Python and numpy through
development of an audio dsp-library.

Author: Johan Ofverstedt
License: BSD
"""

import itertools
import operator
import math
from collections import namedtuple
import numpy as np

def make_buffer(n):
    return np.empty(n, np.double)
def make_zeros(n):
    return np.zeros(n, np.double)
def make_ones(n):
    return np.ones(n, np.double)
def make_delta(n):
    res = make_zeros(n)
    res[0] = 1.0
    return res
def make_noise(n, gain=1.0):
    res = np.random.sample(n)*(2.0*gain) - gain
    return res


def make_range(x):
    return np.arange(x)

def generate(x, f):
    for i in xrange(len(x)):
        x[i] = f()

def clip_copy(x, min=-1.0, max=1.0):
    return x.clip(min, max)
def clip_inplace(x, min=-1.0, max=1.0):
    return x.clip(min, max, x)

class DiscreteCircular:
    def __init__(self, n, init_value=-1):
        self.value = init_value
        self.period = n
        self.wrap_value()

    def __iter__(self):
        return self

    def next(self):
        self.value = self.value + 1
        if self.value == self.period:
            self.value = 0
        return self.value

    def advance(self, n=1):
        self.value += n
        self.wrap_value()

    def frac(self):
        return self.value / float(self.period)

    def wrap_value(self):
        if self.value >= self.period:
            self.value = self.value % self.period
        elif self.value < 0:
            self.value = self.value % self.period        

class SineGenerator:
    def __init__(self, p, phase_shift=0.0):
        self.ci = DiscreteCircular(p)
        self.phase_shift = phase_shift

    def __iter__(self):
        return self

    def __call__(self):
        return self.next()

    def next(self):
        res = math.sin(2.0 * math.pi * self.ci.frac() + self.phase_shift)
        self.ci.next()
        return res

class DelayRange:
    def __init__(self, that, position):
        self.pos = position
        self.buf = that.buf
        while self.pos < 0:
            self.pos += len(self.buf)
        while self.pos >= len(self.buf):
            self.pos += len(self.buf)

    def __iter__(self):
        return self

    def next(self):
        result = self.buf[self.pos]
        if ++self.pos == len(self.buf):
            self.pos = 0
        return result;

class DelayLine:
    def __init__(self, n):
        self.pos = 0
        self.buf = make_zeros(n)

    def read(self, delay):
        return DelayRange(self, pos - delay)
    def write(self, delay):
        return DelayRange(self, pos)

    def rotate(self, n):
        self.pos += n
        while self.pos < 0:
            self.pos += len(self.buf)
        while self.pos >= len(self.buf):
            self.pos += len(self.buf)

BiquadCoeffs = namedtuple('BiquadCoeffs', 'a1 a2 b0 b1 b2')
BiquadState = namedtuple('BiquadState', 'z1 z2')

def main():
    v0 = np.array([1.0, 2.0, 3.0], np.double)
    v1 = np.array([4.0, 5.0, 6.0], np.double)
    print(v0.dtype)
    t = v0+v1
    z = list(t)
    vz = np.empty(17, np.double)
    test = make_ones(16)*2

    d0 = DelayLine(16)

    print(z)
    print(vz)
    print(test)
    print(clip_copy(test))
    print(test)
    print(clip_inplace(test))
    print(test)

    s = make_buffer(16)
    generate(s, SineGenerator(8, 0.5))
    print(s)
    print(make_noise(16, 0.25))
    """print(dot_product(v0, v1))"""
    """print(mean(z))"""

if __name__ == "__main__":
    main()