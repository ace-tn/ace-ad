import torch

class SiteTensor():
    _corner_tensors = None
    _edge_tensors = None
    def __init__(self, dims, init_tensor=None, site_state=[1., 0.], dtype=torch.float64, device=torch.device('cpu')):
        self.dims = dims
        self.dtype = dtype
        self.device = device

        self.initialize_site_tensor(site_state, noise=1e-1)
        self['C'] = self.initialize_corner_tensors()
        self['E'] = self.initialize_edge_tensors()

    def __getitem__(self, key):
        if key == 'A':
            return self._site_tensor
        elif key == 'C':
            return self._corner_tensors
        elif key == 'E':
            return self._edge_tensors
        else:
            raise ValueError(f"Invalid key: '{key}' provided.")

    def __setitem__(self, key, val):
        if key == 'A':
            self._site_tensor = val.to(dtype=self.dtype, device=self.device)
        elif key == 'C':
            self._corner_tensors = [torch.clone(v).to(dtype=self.dtype, device=self.device) for v in val]
        elif key == 'E':
            self._edge_tensors = [torch.clone(v).to(dtype=self.dtype, device=self.device) for v in val]
        else:
            raise ValueError(f"Invalid key: '{key}' provided.")

    def initialize_site_tensor(self, site_state=[1.,], noise=0.0):
        """
        Initializes the site tensor with a given state and optional noise.

        The site tensor is initialized with random values, and the provided `site_state` is added
        to the first entry of the tensor. The tensor is then normalized.

        Args:
            site_state (list, optional): A list specifying the initial state for the site tensor (default is [1.]).
            noise (float, optional): The amount of noise to add to the site tensor (default is 0.0).
        """
        pD = self.dims['phys']
        bD = self.dims['bond']
        self['A'] = noise*torch.rand(bD,bD,bD,bD,pD, dtype=self.dtype, device=self.device)
        for n in range(len(site_state)):
            self['A'][0,0,0,0,n] += site_state[n]
        self['A'] = self['A']/self['A'].norm()

    @staticmethod
    def bond_permutation(k):
        return [(i+k)%4 for i in range(4)] + [4,]

    def bond_permute(self, k):
        """
        Permutes the bond indices of the site tensor.

        This method returns a version of the site tensor with its bond dimensions permuted by an offset `k`.

        Args:
            k (int): The offset used to permute the bond dimensions.

        Returns:
            torch.Tensor: The permuted site tensor.
        """
        return self['A'].permute(self.bond_permutation(k))

    def initialize_corner_tensors(self):
        bD = self.dims['bond']
        init_corner_tensors = []
        directions = 4
        for k in range(directions):
            ak = self['A'].permute(self.bond_permutation(k))
            ck = torch.einsum("lurdp,luRDp->dDrR", ak, ak)
            ck = ck.reshape(bD**2,bD**2)
            ck = ck / ck.norm()
            init_corner_tensors.append(ck)
        return init_corner_tensors

    def initialize_edge_tensors(self):
        bD = self.dims['bond']
        init_edge_tensors = []
        directions = 4
        for k in range(directions):
            ak = self['A'].permute(self.bond_permutation(k))
            ek = torch.einsum("lurdp,LuRDp->lLdDrR", ak, ak)
            ek = ek.reshape(bD**2,bD,bD,bD**2)
            ek = ek / ek.norm()
            init_edge_tensors.append(ek)
        return init_edge_tensors
