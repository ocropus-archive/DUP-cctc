===========================
CCTC
===========================

This is a simple CTC library suitable for use with PyTorch.

Installation
------------

::

    $ python setup.py install


Usage
-----

There are two primary functions, ``cctc_align_targets`` and
``cctc_align_targets_batched``. The first takes a ``length x depth``
tensor, while the second takes a ``batch x length x depth`` tensor
and applies allignment to each batch element.

A call to ``cctc_align_targets(output, source, targets)`` aligns the
source with the targets. The source length must be equal to, or
greater than, the target length. The depths must be equal.Sources and
targets must be normalized along the depth dimension (i.e., represent
posterior probabilities). The output tensor is resized to the proper
dimension and the aligned result returned in the output tensor. Note
that the result is not returned as a return value.

See ``cctc_test.py`` for a simple example of usage.

