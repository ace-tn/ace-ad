import torch
from .rdm import RDM
from acetn.measurement.measure import measure_bond_energy, measure_site_energy
from acetn.model.pauli_matrix import pauli_matrices

def measure(ipeps, model):
    rdm = RDM(ipeps)

    bond_ene = []
    for bond in ipeps.bond_list:
        bond_rdm = rdm[bond]
        bond_norm = torch.einsum("pqpq->", bond_rdm).real
        bond_ene.append(measure_bond_energy(bond, bond_rdm, bond_norm, model))

    X,Y,Z,_ = pauli_matrices(dtype=model.dtype, device=model.device)
    mx = my = mz = 0
    site_ene = []
    for site in ipeps.site_list:
        site_rdm = rdm[site]
        site_norm = torch.einsum("pp->", site_rdm).real
        site_ene.append(measure_site_energy(site, site_rdm, site_norm, model))

        mx += (site_rdm @ X.mat).trace()/site_norm
        my += (site_rdm @ Y.mat).trace()/site_norm
        mz += (site_rdm @ Z.mat).trace()/site_norm

    mx /= len(ipeps.site_list)
    my /= len(ipeps.site_list)
    mz /= len(ipeps.site_list)

    ene = 2*sum(bond_ene)/len(bond_ene) + sum(site_ene)/len(site_ene)
    return ene, mx, my, mz
