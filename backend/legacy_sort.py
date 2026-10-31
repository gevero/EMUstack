"""NumPy 1.24's default argsort (introsort), reproduced for legacy parity.

Legacy EMUstack orders plane waves with np.argsort(key) (default kind), which is not
stable: the order of equal keys depends on the algorithm. Newer NumPy versions use a
different (SIMD) algorithm, so tied plane waves came out in a different order and the
plane-wave matrices (R_net, T_net) were permuted relative to legacy. This is a direct
port of aquicksort_ from numpy/core/src/npysort/quicksort.cpp (NumPy 1.24) for float64.
"""
import numpy as np

_SMALL_QUICKSORT = 15  # npysort_common.h


def _less(a, b):
    return a < b or (b != b and a == a)  # NaNs sort last, as in NumPy


def _aheapsort(v, tosort, start, n):
    # numpy aheapsort_: 1-based heap on tosort[start:start+n]
    a = [None] + [tosort[start + k] for k in range(n)]
    for l in range(n >> 1, 0, -1):
        tmp = a[l]
        i, j = l, l << 1
        while j <= n:
            if j < n and _less(v[a[j]], v[a[j + 1]]):
                j += 1
            if _less(v[tmp], v[a[j]]):
                a[i] = a[j]
                i, j = j, j + j
            else:
                break
        a[i] = tmp
    for n2 in range(n, 1, -1):
        tmp = a[n2]
        a[n2] = a[1]
        i, j = 1, 2
        while j <= n2 - 1:
            if j < n2 - 1 and _less(v[a[j]], v[a[j + 1]]):
                j += 1
            if _less(v[tmp], v[a[j]]):
                a[i] = a[j]
                i, j = j, j + j
            else:
                break
        a[i] = tmp
    for k in range(n):
        tosort[start + k] = a[k + 1]


def legacy_argsort(keys):
    """Indices that sort the real array `keys`, in exactly the order NumPy 1.24's
    np.argsort(keys) returns them (including the order of equal keys)."""
    v = [float(x) for x in np.asarray(keys, dtype=float).ravel()]
    num = len(v)
    tosort = list(range(num))
    if num < 2:
        return np.array(tosort, dtype=np.intp)
    pl, pr = 0, num - 1
    stack, depth = [], []
    cdepth = (num.bit_length() - 1) * 2  # npy_get_msb(num) * 2
    while True:
        if cdepth < 0:
            _aheapsort(v, tosort, pl, pr - pl + 1)
        else:
            while (pr - pl) > _SMALL_QUICKSORT:
                pm = pl + ((pr - pl) >> 1)
                if _less(v[tosort[pm]], v[tosort[pl]]):
                    tosort[pm], tosort[pl] = tosort[pl], tosort[pm]
                if _less(v[tosort[pr]], v[tosort[pm]]):
                    tosort[pr], tosort[pm] = tosort[pm], tosort[pr]
                if _less(v[tosort[pm]], v[tosort[pl]]):
                    tosort[pm], tosort[pl] = tosort[pl], tosort[pm]
                vp = v[tosort[pm]]
                pi, pj = pl, pr - 1
                tosort[pm], tosort[pj] = tosort[pj], tosort[pm]
                while True:
                    pi += 1
                    while _less(v[tosort[pi]], vp):
                        pi += 1
                    pj -= 1
                    while _less(vp, v[tosort[pj]]):
                        pj -= 1
                    if pi >= pj:
                        break
                    tosort[pi], tosort[pj] = tosort[pj], tosort[pi]
                pk = pr - 1
                tosort[pi], tosort[pk] = tosort[pk], tosort[pi]
                # push the larger partition, continue with the smaller one
                if pi - pl < pr - pi:
                    stack.append((pi + 1, pr))
                    pr = pi - 1
                else:
                    stack.append((pl, pi - 1))
                    pl = pi + 1
                cdepth -= 1
                depth.append(cdepth)
            # insertion sort
            for pi in range(pl + 1, pr + 1):
                vi = tosort[pi]
                vp = v[vi]
                pj = pi
                while pj > pl and _less(vp, v[tosort[pj - 1]]):
                    tosort[pj] = tosort[pj - 1]
                    pj -= 1
                tosort[pj] = vi
        if not stack:
            break
        pl, pr = stack.pop()
        cdepth = depth.pop()
    return np.array(tosort, dtype=np.intp)


def z_indexx(beta_z):
    """Legacy z_indexx.f (Numerical Recipes indexx): 0-based indices that sort the
    complex k_z values by increasing -Re(k_z**2), in the order legacy Fortran returns
    them for equal keys. Legacy orders the plane waves of FEM layers with it."""
    z = np.asarray(beta_z, dtype=complex).ravel()
    arr = [-(c.real * c.real - c.imag * c.imag) for c in z]  # -dble(arr(j)**2)
    n = len(arr)
    indx = list(range(n))
    M = 7
    istack = []
    l, ir = 0, n - 1
    while True:
        if ir - l < M:
            for j in range(l + 1, ir + 1):
                indxt = indx[j]
                a = arr[indxt]
                i = j - 1
                while i >= l and arr[indx[i]] > a:
                    indx[i + 1] = indx[i]
                    i -= 1
                indx[i + 1] = indxt
            if not istack:
                break
            l, ir = istack.pop()
        else:
            k = ((l + 1) + (ir + 1)) // 2 - 1  # Fortran k=(l+ir)/2 on 1-based indices
            indx[k], indx[l + 1] = indx[l + 1], indx[k]
            if arr[indx[l]] > arr[indx[ir]]:
                indx[l], indx[ir] = indx[ir], indx[l]
            if arr[indx[l + 1]] > arr[indx[ir]]:
                indx[l + 1], indx[ir] = indx[ir], indx[l + 1]
            if arr[indx[l]] > arr[indx[l + 1]]:
                indx[l], indx[l + 1] = indx[l + 1], indx[l]
            i, j = l + 1, ir
            indxt = indx[l + 1]
            a = arr[indxt]
            while True:
                i += 1
                while arr[indx[i]] < a:
                    i += 1
                j -= 1
                while arr[indx[j]] > a:
                    j -= 1
                if j < i:
                    break
                indx[i], indx[j] = indx[j], indx[i]
            indx[l + 1] = indx[j]
            indx[j] = indxt
            # Fortran: if (ir-i+1 >= j-l) push (i, ir), else push (l, j-1); same in 0-based
            if ir - i + 1 >= j - l:
                istack.append((i, ir))
                ir = j - 1
            else:
                istack.append((l, j - 1))
                l = i
    return np.array(indx, dtype=np.intp)


def grating_orders_2d(max_order, dx, dy):
    """Grating orders (px, py) kept for a 2D lattice of periods dx, dy, in legacy order
    (px in the outer loop, py in the inner one).

    An order is kept when (px/dx)**2 + (py/dy)**2 <= (max_order/max(dx, dy))**2. Legacy
    evaluated this in nm for the thin films and in units of the period for the FEM
    layers, so rounding could keep an order lying exactly on the boundary in one and
    drop it in the other (e.g. (3, 4) for max_order 5 with a 480 nm period), and the
    run stopped because the layers had different numbers of plane waves. Here the test
    is done in units of the longer period, with a tiny tolerance, so the result does not
    depend on the units and boundary orders are always kept. Wherever legacy could run,
    this keeps the same orders as legacy.
    """
    p = np.arange(-max_order, max_order + 1)
    pys, pxs = np.meshgrid(p, p)
    longest = max(dx, dy)
    radius2 = (pxs * (longest / dx))**2 + (pys * (longest / dy))**2
    keep = radius2 <= max_order**2 * (1.0 + 1e-12)
    return pxs[keep], pys[keep]
