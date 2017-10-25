from pylab import *
import sys
import torch
import cctc
a = torch.randn(3, 3)
print a
cctc.square(a)
print a

def rownorm(t):
  return t / t.sum(1).repeat(1,t.size(1))
def batch_rownorm(t):
  for i in range(len(t)):
    t.select(0, i).copy_(rownorm(t.select(0, i)))
  return t

a = rownorm(torch.rand(100, 17))
b = rownorm(torch.rand(20, 17))
c = torch.rand(1, 1)
print c
cctc.ctc_align_targets(c, a, b)
print c

a = batch_rownorm(torch.rand(3, 100, 17))
b = batch_rownorm(torch.rand(3, 20, 17))
c = torch.rand(1)
print a.size(), b.size(), c.size()
print c
cctc.ctc_align_targets_batch(c, a, b)
print c
