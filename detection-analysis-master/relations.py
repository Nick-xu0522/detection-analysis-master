# encoding: utf-8
import numpy as np

def _context_ralation(trace):  # behavioral context
    """
    Extacting behavior context relations from trace.
    """
    res = {}
    for i in range(1,len(trace) - 1):
        name = '<%s>,<%s>' % (trace[i - 1], trace[i + 1])
        flag = True
        for key in res.keys():
            ab = key.split(",")
            if (trace[i - 1] == ab[0] and trace[i + 1] == ab[2]):
                del res[key]
                flag = False
                break
        if flag:
            res[name] = 1

    return res


def transform(traces, types):
    """
    transform traces into relation matrix
    """
    n_traces = len(traces)
    matrixs = {}
    for relation_type in types:
        matrix = {}
        for i, trace in enumerate(traces):
            for key, value in relation_type['extract'](trace).items():
                array = matrix.setdefault(key, np.full(n_traces, relation_type['default'], np.int8))
                array[i] = value
        matrixs[relation_type['name']] = matrix
    return matrix


CONTEXT_RALATION = {
    'name': 'context_ralation',
    'extract': _context_ralation,
    'default': 0,
}


