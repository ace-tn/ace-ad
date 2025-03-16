import torch
from acead.linalg.svd import SVD
svd = SVD.apply

def make_quarter_tensor(ak, ck, ek1, ek2):
    bD = ak.shape[1]
    cD1 = ek1.shape[0]
    cD2 = ek2.shape[3]
    q1 = torch.einsum("ab,clLa->bclL", ck, ek1)
    q1 = torch.einsum("bclL,buUe->celLuU", q1, ek2)
    q1 = torch.einsum("celLuU,lurdp->ceLUprd", q1, ak)
    q1 = torch.einsum("ceLUprd,LURDp->cdDerR", q1, ak)
    q1 = q1.reshape(cD1*bD*bD, cD2*bD*bD)
    q1 = q1 / q1.norm()
    return q1


def calculate_projector_1(r2, vf0, sf0, cD, bD, cD_new):
    proj1 = r2 @ vf0[:,:cD_new]
    proj1 = proj1 * torch.sqrt(1/sf0[:cD_new])
    proj1 = proj1.reshape(cD, bD, bD, cD_new)
    return proj1


def calculate_projector_2(r1, uf0, sf0, cD, bD, cD_new):
    proj2 = uf0[:,:cD_new].T @ r1
    proj2 = proj2.T * torch.sqrt(1/sf0[:cD_new])
    proj2 = proj2.reshape(cD, bD, bD, cD_new)
    return proj2


def bond_permutation(k):
    return [(i+k)%4 for i in range(4)] + [4,]


def calculate_projectors(tensors, ipeps, sites, k):
    a_tensors, c_tensors, e_tensors = tensors
    s1, s2, s3, s4 = sites

    a1k = a_tensors[s1].permute(bond_permutation(k))
    c1  = c_tensors[s1][k]
    e11 = e_tensors[s1][(k+3)%4]
    e12 = e_tensors[s1][k]
    q1 = make_quarter_tensor(a1k, c1, e11, e12)

    a2k = a_tensors[s2].permute(bond_permutation((k+1)%4))
    c2  = c_tensors[s2][(k+1)%4]
    e21 = e_tensors[s2][k]
    e22 = e_tensors[s2][(k+1)%4]
    q2 = make_quarter_tensor(a2k, c2, e21, e22)

    a3k = a_tensors[s3].permute(bond_permutation((k+2)%4))
    c3  = c_tensors[s3][(k+2)%4]
    e31 = e_tensors[s3][(k+1)%4]
    e32 = e_tensors[s3][(k+2)%4]
    q3 = make_quarter_tensor(a3k, c3, e31, e32)

    a4k = a_tensors[s4].permute(bond_permutation((k+3)%4))
    c4  = c_tensors[s4][(k+3)%4]
    e41 = e_tensors[s4][(k+2)%4]
    e42 = e_tensors[s4][(k+3)%4]
    q4 = make_quarter_tensor(a4k, c4, e41, e42)

    r1 = q4 @ q1
    r2 = q2 @ q3
    f0 = r1 @ r2

    uf0,sf0,vf0 = svd(f0)

    # Calculate projectors
    bD = ipeps.dims['bond']
    cD_max = ipeps.dims['chi']
    cD_sf0 = sum(abs(sf0/sf0[0]) > 1e-12)
    cD_new = max(bD*bD, min(cD_sf0, cD_max))

    cD1 = e12.shape[3]
    proj1_i = calculate_projector_1(r2, vf0, sf0, cD1, bD, cD_new)

    cD2 = e21.shape[0]
    proj2_i = calculate_projector_2(r1, uf0, sf0, cD2, bD, cD_new)

    return proj1_i, proj2_i


def renormalize_ci1(cj, ej, proj):
    ci = torch.einsum("ab,clLa->cblL", cj, ej)
    ci = torch.einsum("cblL,blLe->ce", ci, proj)
    return ci / ci.norm()


def renormalize_ci2(cj, ej, proj):
    ci = torch.einsum("ab,brRc->arRc", cj, ej)
    ci = torch.einsum("arRc,arRf->fc", ci, proj)
    return ci / ci.norm()


def renormalize_ei(aj, ej, proj1, proj2):
    ei = torch.einsum("alLe,auUc->lLeuUc", proj2, ej)
    ei = torch.einsum("lLeuUc,lurdp->LeUcprd", ei, aj)
    ei = torch.einsum("LeUcprd,LURDp->ecrdRD", ei, aj)
    ei = torch.einsum("ecrdRD,crRf->edDf", ei, proj1)
    return ei / ei.norm()


def renormalize_boundary(ipeps, proj1, proj2, s1, s2, i, j, k):
    ci = ipeps[s1]['C'][(3+k)%4]
    ei = ipeps[s1]['E'][(2+k)%4]
    ipeps[s2]['C'][(3+k)%4] = renormalize_ci1(ci, ei, proj1[i])

    ci = ipeps[s1]['C'][k]
    ei = ipeps[s1]['E'][k]
    ipeps[s2]['C'][k] = renormalize_ci2(ci, ei, proj2[j])

    ai = ipeps[s1].bond_permute(k)
    ei = ipeps[s1]['E'][(3+k)%4]
    ipeps[s2]['E'][(3+k)%4] = renormalize_ei(ei, ai, proj2[i], proj1[j])


def up_absorb(tensors, ipeps, yi):
    a_tensors, c_tensors, e_tensors = tensors

    proj1 = []
    proj2 = []
    for xi in range(ipeps.nx):
        xj = (xi+1) % ipeps.nx
        yj = (yi+1) % ipeps.ny
        s1 = (xi,yj)
        s2 = (xj,yj)
        s3 = (xj,yi)
        s4 = (xi,yi)
        sites = [s1,s2,s3,s4]
        proj1_i, proj2_i = calculate_projectors(tensors, ipeps, sites, k=0)
        proj1.append(proj1_i)
        proj2.append(proj2_i)

    c1_new = []
    c2_new = []
    e_new = []
    for xi in range(ipeps.nx):
        xj = (xi+ipeps.nx-1) % ipeps.nx
        yj = (yi+1) % ipeps.ny
        aj = a_tensors[(xi,yj)]
        cj = c_tensors[(xi,yj)]
        ej = e_tensors[(xi,yj)]

        # Renormalize environment tensors
        c1_new.append(renormalize_ci1(cj[0], ej[3], proj1[xj]))
        c2_new.append(renormalize_ci2(cj[1], ej[1], proj2[xi]))
        e_new.append(renormalize_ei(aj, ej[0], proj1[xi], proj2[xj]))

    return c1_new, c2_new, e_new


def up_move(a_tensors, c_tensors, e_tensors, ipeps):
    for yi in range(ipeps.ny-1,-1,-1):
        c1_new, c2_new, e_new = up_absorb([a_tensors, c_tensors, e_tensors], ipeps, yi)
        for xi in range(ipeps.nx):
            c_tensors[(xi,yi)][0] = c1_new[xi]
            c_tensors[(xi,yi)][1] = c2_new[xi]
            e_tensors[(xi,yi)][0] = e_new[xi]
            ipeps[(xi,yi)]['C'] = c_tensors[(xi,yi)]
            ipeps[(xi,yi)]['E'] = e_tensors[(xi,yi)]


def right_absorb(tensors, ipeps, xi):
    a_tensors, c_tensors, e_tensors = tensors

    proj1 = []
    proj2 = []
    for yi in range(ipeps.ny):
        xj = (xi+1) % ipeps.nx
        yj = (yi+1) % ipeps.ny
        s1 = (xj,yj)
        s2 = (xj,yi)
        s3 = (xi,yi)
        s4 = (xi,yj)
        sites = [s1,s2,s3,s4]
        proj1_i, proj2_i = calculate_projectors(tensors, ipeps, sites, k=1)
        proj1.append(proj1_i)
        proj2.append(proj2_i)

    c1_new = []
    c2_new = []
    e_new = []
    for yi in range(ipeps.ny):
        xj = (xi+1) % ipeps.nx
        yj = (yi+ipeps.ny-1) % ipeps.ny
        aj = a_tensors[(xj,yi)].permute(bond_permutation(1))
        cj = c_tensors[(xj,yi)]
        ej = e_tensors[(xj,yi)]

        # Renormalize boundary tensors
        c1_new.append(renormalize_ci1(cj[1], ej[0], proj1[yi]))
        c2_new.append(renormalize_ci2(cj[2], ej[2], proj2[yj]))
        e_new.append(renormalize_ei(aj, ej[1], proj1[yj], proj2[yi]))

    return c1_new, c2_new, e_new


def right_move(a_tensors, c_tensors, e_tensors, ipeps):
    for xi in range(ipeps.nx-1,-1,-1):
        c1_new, c2_new, e_new = right_absorb([a_tensors, c_tensors, e_tensors], ipeps, xi)
        for yi in range(ipeps.ny):
            c_tensors[(xi,yi)][1] = c1_new[yi]
            c_tensors[(xi,yi)][2] = c2_new[yi]
            e_tensors[(xi,yi)][1] = e_new[yi]
            ipeps[(xi,yi)]['C'] = c_tensors[(xi,yi)]
            ipeps[(xi,yi)]['E'] = e_tensors[(xi,yi)]


def down_absorb(tensors, ipeps, yi):
    a_tensors, c_tensors, e_tensors = tensors

    proj1 = []
    proj2 = []
    for xi in range(ipeps.nx):
        xj = (xi+1) % ipeps.nx
        yj = (yi+ipeps.ny-1) % ipeps.ny
        s1 = (xj,yj)
        s2 = (xi,yj)
        s3 = (xi,yi)
        s4 = (xj,yi)
        sites = [s1,s2,s3,s4]
        proj1_i, proj2_i = calculate_projectors(tensors, ipeps, sites, k=2)
        proj1.append(proj1_i)
        proj2.append(proj2_i)

    c1_new = []
    c2_new = []
    e_new = []
    for xi in range(ipeps.nx):
        xj = (xi+ipeps.nx-1) % ipeps.nx
        yj = (yi+ipeps.ny-1) % ipeps.ny
        aj = a_tensors[(xi,yj)].permute(bond_permutation(2))
        cj = c_tensors[(xi,yj)]
        ej = e_tensors[(xi,yj)]

        # Renormalize environment tensors
        c1_new.append(renormalize_ci1(cj[2], ej[1], proj1[xi]))
        c2_new.append(renormalize_ci2(cj[3], ej[3], proj2[xj]))
        e_new.append(renormalize_ei(aj, ej[2], proj1[xj], proj2[xi]))

    return c1_new, c2_new, e_new


def down_move(a_tensors, c_tensors, e_tensors, ipeps):
    for yi in range(ipeps.ny):
        c1_new, c2_new, e_new = down_absorb([a_tensors, c_tensors, e_tensors], ipeps, yi)
        for xi in range(ipeps.nx):
            c_tensors[(xi,yi)][2] = c1_new[xi]
            c_tensors[(xi,yi)][3] = c2_new[xi]
            e_tensors[(xi,yi)][2] = e_new[xi]
            ipeps[(xi,yi)]['C'] = c_tensors[(xi,yi)]
            ipeps[(xi,yi)]['E'] = e_tensors[(xi,yi)]


def left_absorb(tensors, ipeps, xi):
    a_tensors, c_tensors, e_tensors = tensors

    proj1 = []
    proj2 = []
    for yi in range(ipeps.ny):
        xj = (xi+ipeps.nx-1) % ipeps.nx
        yj = (yi+1) % ipeps.ny
        s1 = (xj,yi)
        s2 = (xj,yj)
        s3 = (xi,yj)
        s4 = (xi,yi)
        sites = [s1,s2,s3,s4]
        proj1_i, proj2_i = calculate_projectors(tensors, ipeps, sites, k=3)
        proj1.append(proj1_i)
        proj2.append(proj2_i)

    c1_new = []
    c2_new = []
    e_new = []
    for yi in range(ipeps.ny):
        xj = (xi+ipeps.nx-1) % ipeps.nx
        yj = (yi+ipeps.ny-1) % ipeps.ny
        aj = a_tensors[(xj,yi)].permute(bond_permutation(3))
        cj = c_tensors[(xj,yi)]
        ej = e_tensors[(xj,yi)]

        # Renormalize environment tensors
        c1_new.append(renormalize_ci1(cj[3], ej[2], proj1[yj]))
        c2_new.append(renormalize_ci2(cj[0], ej[0], proj2[yi]))
        e_new.append(renormalize_ei(aj, ej[3], proj1[yi], proj2[yj]))

    return c1_new, c2_new, e_new


def left_move(a_tensors, c_tensors, e_tensors, ipeps):
    for xi in range(ipeps.nx):
        c1_new, c2_new, e_new = left_absorb([a_tensors, c_tensors, e_tensors], ipeps, xi)
        for yi in range(ipeps.ny):
            c_tensors[(xi,yi)][3] = c1_new[yi]
            c_tensors[(xi,yi)][0] = c2_new[yi]
            e_tensors[(xi,yi)][3] = e_new[yi]
            ipeps[(xi,yi)]['C'] = c_tensors[(xi,yi)]
            ipeps[(xi,yi)]['E'] = e_tensors[(xi,yi)]


class CTMRG(torch.nn.Module):
    def __init__(self, ipeps):
        super().__init__()
        self.ipeps = ipeps
        self.steps = ipeps.ctmrg_steps
        self.a_tensors = {}
        self.c_tensors = {}
        self.e_tensors = {}
        for xi in range(ipeps.nx):
            for yi in range(ipeps.ny):
                self.a_tensors[(xi,yi)] = ipeps[(xi,yi)]['A'].clone() / ipeps[(xi,yi)]['A'].norm()
                self.c_tensors[(xi,yi)] = [c.clone() for c in ipeps[(xi,yi)]['C']]
                self.e_tensors[(xi,yi)] = [e.clone() for e in ipeps[(xi,yi)]['E']]

    def forward(self, ipeps):
        for _ in range(self.steps):
            up_move(self.a_tensors, self.c_tensors, self.e_tensors, ipeps)
            down_move(self.a_tensors, self.c_tensors, self.e_tensors, ipeps)
            right_move(self.a_tensors, self.c_tensors, self.e_tensors, ipeps)
            left_move(self.a_tensors, self.c_tensors, self.e_tensors, ipeps)

        for xi in range(self.ipeps.nx):
            for yi in range(self.ipeps.ny):
                ipeps[(xi,yi)]['C'] = self.c_tensors[(xi,yi)]
                ipeps[(xi,yi)]['E'] = self.e_tensors[(xi,yi)]
