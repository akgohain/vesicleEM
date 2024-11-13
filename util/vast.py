import numpy as np


def read_vast_seg(fn):
    a = open(fn).readlines()
    # remove comments
    st_id = 0
    while a[st_id][0] in ["%", "\\", '\n']:
        st_id += 1
    
    st_id -= 1
    # remove segment name
    out = np.zeros((len(a) - st_id - 1, 24), dtype=int)
    name = [None] * (len(a) - st_id - 1)
    for i in range(st_id + 1, len(a)):
        out[i - st_id - 1] = np.array(
            [int(x) for x in a[i][: a[i].find('"')].split(" ") if len(x) > 0]
        )
        name[i - st_id - 1] = a[i][a[i].find('"') + 1 : a[i].rfind('"')]
    return out, name

def vast_meta_relabel(
    fn, kw_bad=["bad", "del"], kw_nm=["merge"], do_print=False
):
    # if there is meta data
    print("load meta")
    dd, nn = read_vast_seg(fn)
    rl = np.arange(1 + dd.shape[0], dtype=np.uint16)
    pid = np.unique(dd[:, 13])
    if do_print:
        print(
            ",".join(
                [
                    nn[x]
                    for x in np.where(np.in1d(dd[:, 0], pid))[0]
                    if "Imported Segment" not in nn[x]
                ]
            )
        )

    pid_b = []
    if len(kw_bad) > 0:
        # delete seg id
        pid_b = [
            i
            for i, x in enumerate(nn)
            if max([y in x.lower() for y in kw_bad])
        ]
        bid = np.where(np.in1d(dd[:, 13], pid_b))[0]
        bid = np.hstack([pid_b, bid])
        if len(bid) > 0:
            rl[bid] = 0
        print("found %d bad" % (len(bid)))

    # not to merge
    kw_nm += ["background"]
    pid_nm = [
        i for i, x in enumerate(nn) if max([y in x.lower() for y in kw_nm])
    ]
    # pid: all are children of background
    pid_nm = np.hstack([pid_nm, pid_b])
    print("apply meta")
    # hierarchical
    for p in pid[np.in1d(pid, pid_nm, invert=True)]:
        rl[dd[dd[:, 13] == p, 0]] = p
    # consolidate root labels
    for u in np.unique(rl[np.where(rl[rl] != rl)[0]]):
        u0 = u
        while u0 != rl[u0]:
            rl[rl == u0] = rl[u0]
            u0 = rl[u0]
        print(u, rl)
    return rl