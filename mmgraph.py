import time
import math
import numpy as np
import itertools
import matplotlib.pyplot as plt

import Orange
import Orange.data.filter
import Orange.data.continuizer

from Orange.statistics import contingency

R = 1

def std(f):
    x = np.array(range(len(f)))
    # normalize; we do not prefer attributes with many values
    x = x / x.mean()
    xf = np.multiply(f, x)
    x2f = np.multiply(f, np.power(x, 2))
    return np.sqrt((np.sum(x2f) - np.power(np.sum(xf), 2) / np.sum(f)) / (np.sum(f) - 1))


def p_index_ct(ct):
    """Projection pursuit projection index."""
    ni, nj = ct.shape

    # compute standard deviation
    s = std(np.sum(ct, axis=1)) * std(np.sum(ct, axis=0))

    pairs = [(v1, v2) for v1 in range(ni) for v2 in range(nj)]

    d = sum(ct[pairs[p1]] * ct[pairs[p2]] * max(1.4142135623730951 - np.sqrt(
        np.power((pairs[p1][0] - pairs[p2][0]) / float(ni - 1), 2) + np.power(
            (pairs[p1][1] - pairs[p2][1]) / float(nj - 1), 2)), 0.) for p1 in range(len(pairs)) for p2 in range(p1))

    ssum = len(pairs) * (len(pairs) - 1) / 2.

    return s * d / ssum, s, d / ssum


def p_index(t, a1, a2):
    #print(a1.name)
    a1_mean = np.mean([ex[a1] for ex in t])
    a1_s = math.sqrt(sum(pow(ex[a1] - a1_mean, 2) for ex in t) / float(len(t)))
    #print("a1 s", a1_s)

    #print(a2.name)
    a2_mean = np.mean([ex[a2] for ex in t])
    a2_s = math.sqrt(sum(pow(ex[a2] - a2_mean, 2) for ex in t) / float(len(t)))
    #print("a2 s", a2_s)

    s = a1_s * a2_s
    #print("s", s)

    #a1_d = sum(R - abs(t[i][a1] - t[j][a1]) for i in range(len(t)) for j in range(i))
    #print("a1 d", a1_d)

    #a2_d = sum(R - abs(t[i][a2] - t[j][a2]) for i in range(len(t)) for j in range(i))
    #print("a2 d", a2_d)

    # for i in range(len(t)):
    #     for j in range(i):
    #         print(t[i][a1] - t[j][a1])

    d = sum(R - math.sqrt(pow(t[i][a1] - t[j][a1], 2) + pow(t[i][a2] - t[j][a2], 2)) for i in range(len(t)) for j in range(i))
    #print("d", d)
    #print("s*d", s * d)
    return s * d


def p_index_ct_manhattan(ct):
    """Projection pursuit projection index with Manhattan distance."""
    ni, nj = ct.shape
    norm_i = float(ni - 1)
    norm_j = float(nj - 1)

    # compute standard deviation
    s = std(np.sum(ct, axis=1)) * std(np.sum(ct, axis=0))

    pairs_row = [(v1, v2) for v1 in range(ni) for v2 in range(ni)]
    pairs_col = [(v1, v2) for v1 in range(nj) for v2 in range(nj)]

    sum_row = np.sum(ct, axis=1)
    sum_col = np.sum(ct, axis=0)

    d_row = sum(sum_row[p1] * sum_row[p2] * (1 - np.abs(p1 - p2) / norm_i) for p1, p2 in pairs_row)
    d_col = sum(sum_col[p1] * sum_col[p2] * (1 - np.abs(p1 - p2) / norm_j) for p1, p2 in pairs_col)

    d = d_row / len(pairs_row) + d_col / len(pairs_col)

    return s * d


def randomize_ct(ct):
    """"Randomize contingency table by keeping distribution of corresponding attributes."""
    d0 = np.sum(ct, axis=1)
    d1 = np.sum(ct, axis=0)
    n0 = np.sum(d0)
    n1 = np.sum(d1)

    assert n1 == n0

    draw0 = list(itertools.chain(*([i] * j for i, j in enumerate(np.random.multinomial(n0, d0 / n0)))))
    draw1 = list(itertools.chain(*([i] * j for i, j in enumerate(np.random.multinomial(n0, d1 / n0)))))
    random.shuffle(draw1)
    points = zip(draw0, draw1)

    ct_ = np.zeros((len(d0), len(d1)))
    for p in points:
        ct_[p] += 1

    return ct_


def p_value(ct, sample_size=1000, p_index_func=p_index_ct_manhattan):
    """Compute p-value of projection index score."""
    pindex = p_index_func(ct)
    pindexes = (p_index_func(randomize_ct(ct)) for _ in range(sample_size))
    pvalue = len([p for p in pindexes if p > pindex]) / float(sample_size)

    return pvalue, pindex

#wine = Orange.data.sql.table.SqlTable(host='localhost', database='test', table='wine_1000')

dataset = 'WDBC'
cachefile = '{0}.pkl'.format(dataset)

odata = Orange.data.Table(dataset)
#print(len(odata), len(odata.domain.attributes), len([a for a in odata.domain if type(a) == Orange.data.variable.ContinuousVariable]))

Ns = []
times1 = []
times2 = []
for N in range(20, len(odata) + 20, 20):
    N = min(N, len(odata))
    data = odata[:N]

    newdom = Orange.data.continuizer.DomainContinuizer(data, normalize_continuous=Orange.data.continuizer.DomainContinuizer.NormalizeBySpan, zero_based=True)
    cdata = Orange.data.Table(newdom, data)

    disc = Orange.feature.discretization.EqualWidth(n=10)
    newdom = Orange.data.Domain(
        [disc(data, attr) if type(attr) == Orange.data.variable.ContinuousVariable
         else attr for attr in data.domain.attributes], data.domain.class_vars)
    ddata = Orange.data.Table(newdom, data)

    print('N', len(cdata), len(ddata), len(data))
    print('attributes', len(cdata.domain), len(ddata.domain), len(data.domain))
    print('continious attributes', len([a for a in data.domain if type(a) == Orange.data.variable.ContinuousVariable]))

    #print([a for a in cdata.domain])
    #data = Orange.data.Table(ndomain, data)
    #

    print("COMPUTING SCORES for {0:0.0f} scatter plots".format(
        len(cdata.domain.attributes) * (len(cdata.domain.attributes) - 1) / 2))

    cattrs = cdata.domain.attributes
    dattrs = ddata.domain.attributes

    print(cattrs)
    print(dattrs)

    t0 = time.time()
    for i in range(len(cattrs)):
        for j in range(i):
            print(p_index(cdata, cattrs[i], cattrs[j]))

    t1 = time.time() - t0
    print('t1', t1)

    t0 = time.time()
    for i in range(len(dattrs)):
        for j in range(i):
            ct = np.array(contingency.get_contingency(ddata, dattrs[j], dattrs[i]))
            print(p_index_ct(ct))

    t2 = time.time() - t0
    print('t2', t2)

    Ns.append(N)
    times1.append(t1)
    times2.append(t2)

    fig, ax = plt.subplots()
    ax.plot(Ns, times1, '-', Ns, times2, '-')
    ax.set_xlabel('Number of examples')
    ax.set_ylabel('Time in seconds')
    ax.set_title('Rank Scatterplots - {}'.format(dataset))
    plt.xlim(min(Ns), max(Ns))
    plt.legend(('Original', 'Contingency table'), 'upper left')
    plt.tight_layout()
    plt.savefig('rank-{}.pdf'.format(dataset.lower()))

print(Ns)
print(times1)
print(times2)
