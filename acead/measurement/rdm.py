from torch import einsum, conj

class RDM:
    """
    A class that computes the reduced density matrix (RDM) for a given iPEPS tensor network.

    The `RDM` class contains methods to build site and bond RDMs by contracting the appropriate tensors from the iPEPS structure.
    The reduced density matrix represents a subregion of the system and is used in calculating measurements.

    Attributes:
        ipeps (object): The iPEPS object containing tensors for site and bond indices.
    """
    def __init__(self, ipeps):
        """
        Initializes the `RDM` object with a given iPEPS tensor network.

        Args:
            ipeps (object): The iPEPS object containing tensors for the iPEPS network.
        """
        self.ipeps = ipeps

    def __getitem__(self, key):
        """
        Accessor method for building reduced density matrices (RDMs).

        Args:
            key (tuple or list): 
                - A tuple of two elements (site indices) to build a site RDM.
                - A list of three elements (bond indices) to build a bond RDM.

        Returns:
            torch.Tensor: The reduced density matrix corresponding to the provided indices (site or bond).
        """
        if isinstance(key, tuple) and len(key) == 2:
            return self.build_site_rdm(key)
        if isinstance(key, list) and len(key) == 3:
            return self.build_bond_rdm(key)

    def build_site_rdm(self, site):
        """
        Builds the reduced density matrix (RDM) for a single site in the iPEPS network.

        This method constructs the site RDM by contracting various tensors from the iPEPS object 
        (site tensor 'A', corner tensors 'C', and edge tensors 'E') according to the specified procedure.

        Args:
            site (tuple): Indices for the site (e.g., a pair of indices in the iPEPS network).

        Returns:
            torch.Tensor: The reduced density matrix (RDM) for the specified site.
        """
        c1 = self.ipeps[site]['C'][0]
        c2 = self.ipeps[site]['C'][1]
        c3 = self.ipeps[site]['C'][2]
        c4 = self.ipeps[site]['C'][3]
        e1 = self.ipeps[site]['E'][0]
        e2 = self.ipeps[site]['E'][1]
        e3 = self.ipeps[site]['E'][2]
        e4 = self.ipeps[site]['E'][3]
        a1 = self.ipeps[site]['A']

        tmp1 = einsum("ab,blLc->aclL", c4, e4)
        tmp1 = einsum("aclL,edDa->clLedD", tmp1, e3)
        tmp1 = einsum("clLedD,LURDP->cledURP", tmp1, conj(a1))
        tmp2 = einsum("ab,buUc->acuU", c1, e1)
        tmp3 = einsum("ab,crRa->bcrR", c3, e2)
        tmp3 = einsum("ec,bcrR->ebrR", c2, tmp3)
        tmp3 = einsum("ebrR,aeuU->brRauU", tmp3, tmp2)
        tmp3 = einsum("erRcuU,cledURP->ruldP", tmp3, tmp1)
        rdm = einsum("ruldP,lurdp->Pp", tmp3, a1)
        return rdm

    def build_bond_rdm(self, bond):
        """
        Builds the reduced density matrix (RDM) for a bond in the iPEPS network.

        This method constructs the bond RDM by contracting tensors from two neighboring sites (bond indices) 
        in the iPEPS network and combining them according to a specific contraction scheme.

        Args:
            bond (list): A list containing three elements, where the first two are site indices and the third is the bond direction.

        Returns:
            torch.Tensor: The reduced density matrix (RDM) for the specified bond.
        """
        s1,s2,k = bond

        c12 = self.ipeps[s1]['C'][(k+1)%4]
        e12 = self.ipeps[s1]['E'][(k+1)%4]
        e11 = self.ipeps[s1]['E'][(k+0)%4]
        c13 = self.ipeps[s1]['C'][(k+2)%4]
        e13 = self.ipeps[s1]['E'][(k+2)%4]
        a1  = self.ipeps[s1].bond_permute(k)

        temp1 = einsum("ab,brRe->arRe", c12, e12)
        temp1 = einsum("arRe,cuUa->ecuUrR", temp1, e11)
        temp1 = einsum("bcuUrR,lurdq->qbclURd", temp1, a1)
        temp1 = einsum("qbclURd,LURDQ->qQbclLdD", temp1, conj(a1))
        temp2 = einsum("ab,bdDc->adDc", c13, e13)
        rdm1_tmp = einsum("qQbclLdD,bdDe->qQcelL", temp1, temp2)

        c21 = self.ipeps[s2]['C'][(k+0)%4]
        e21 = self.ipeps[s2]['E'][(k+0)%4]
        e24 = self.ipeps[s2]['E'][(k+3)%4]
        c24 = self.ipeps[s2]['C'][(k+3)%4]
        e23 = self.ipeps[s2]['E'][(k+2)%4]
        a2  = self.ipeps[s2].bond_permute(k)

        temp1 = einsum("ab,buUc->auUc", c21, e21)
        temp1 = einsum("alLb,buUc->acuUlL", e24, temp1)
        temp1 = einsum("acuUlL,lurdp->pacULrd", temp1, a2)
        temp1 = einsum("pacULrd,LURDP->pPacdDrR", temp1, conj(a2))
        temp2 = einsum("ab,cdDa->bdDc", c24, e23)
        rdm2_tmp = einsum("adDe,pPacdDrR->pPcerR", temp2, temp1)

        rdm = einsum("pPcelL,qQcelL->pqPQ", rdm1_tmp, rdm2_tmp)
        return rdm
