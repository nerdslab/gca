import torch
import torch.nn as nn
import torch.nn.functional as F


class BaseLoss(nn.Module):
    def __init__(self, batch_size, temperature, iterations=10, lam=None, q=None):
        """
        Base class for loss functions.

        Args:
            batch_size (int): Number of samples per batch.
            temperature (float): Temperature parameter for scaling similarities.
            iterations (int, optional): Number of iterations for Sinkhorn normalization. Defaults to 10.
            lam (float, optional): Lambda parameter for weighting. Defaults to None.
            q (float, optional): Q parameter for generalized cross-entropy. Defaults to None.
        """
        super(BaseLoss, self).__init__()
        self.batch_size = batch_size
        self.temperature = temperature
        self.iterations = iterations
        self.lam = lam
        self.q = q
        self.similarity_f = nn.CosineSimilarity(dim=2)

    def mask_correlated_samples(self, batch_size):
        """
        Creates a mask to identify positive and negative samples.

        Args:
            batch_size (int): Number of samples per batch.

        Returns:
            torch.Tensor: Boolean mask tensor.
        """
        N = 2 * batch_size
        mask = torch.ones((N, N), dtype=bool, device=self._get_device())
        mask = mask.fill_diagonal_(0)
        for i in range(batch_size):
            mask[i, batch_size + i] = 0
            mask[batch_size + i, i] = 0
        return mask

    def compute_cost(self, z):
        """
        Computes the cost matrix based on embeddings.

        Args:
            z (torch.Tensor): Embedding tensor.

        Returns:
            torch.Tensor: Cost matrix.
        """
        raise NotImplementedError("compute_cost must be implemented by the subclass.")

    def gibs_kernel(self, z):
        """
        Computes the Gibbs kernel based on the cost matrix.

        Args:
            z (torch.Tensor): Embedding tensor.

        Returns:
            torch.Tensor: Gibbs kernel.
        """
        Cost = self.compute_cost(z)
        kernel = torch.exp(-Cost / self.temperature)
        return kernel

    def forward(self, z):
        """
        Forward pass to compute the loss.

        Args:
            z (torch.Tensor): Embedding tensor.

        Returns:
            torch.Tensor: Computed loss.
        """
        raise NotImplementedError("forward must be implemented by the subclass.")

    def _get_device(self):
        """
        Retrieves the device of the model parameters.

        Returns:
            torch.device: Current device.
        """
        return next(self.parameters()).device if len(list(self.parameters())) > 0 else torch.device('cpu')

class NT_Xent(BaseLoss):
    def __init__(self, batch_size, temperature, iterations=10, lam=None, q=None):
        """
        NT-Xent loss for contrastive learning.

        Args:
            batch_size (int): Number of samples per batch.
            temperature (float): Temperature parameter for scaling similarities.
            iterations (int, optional): Number of Sinkhorn iterations. Defaults to 10.
            lam (float, optional): Lambda parameter. Not used here. Defaults to None.
            q (float, optional): Q parameter. Not used here. Defaults to None.
        """
        super(NT_Xent, self).__init__(batch_size, temperature, iterations, lam, q)
        self.mask = self.mask_correlated_samples(batch_size)
        self.criterion = nn.CrossEntropyLoss(reduction="sum")

    def compute_cost(self, z):
        """
        Computes the cost matrix based on cosine similarity.

        Args:
            z (torch.Tensor): Normalized embedding tensor.

        Returns:
            torch.Tensor: Cost matrix.
        """
        z = F.normalize(z, dim=1)
        z_scores = (z @ z.t()).clamp(min=1e-7)
        Cost = 1 - z_scores
        C = Cost.max()
        diags = C * torch.eye(Cost.shape[0]).to(Cost.device)
        diagsoff = (1 - torch.eye(Cost.shape[0]).to(Cost.device))* Cost
        Cost = diags + diagsoff
        return Cost




    def forward(self, z):
        """
        Forward pass for NT-Xent loss.

        Args:
            z (torch.Tensor): Embedding tensor.

        Returns:
            torch.Tensor: Computed NT-Xent loss.
        """
        N = 2 * self.batch_size
        sim = self.similarity_f(z.unsqueeze(1), z.unsqueeze(0)) / self.temperature
        sim_i_j = torch.diag(sim, self.batch_size)
        sim_j_i = torch.diag(sim, -self.batch_size)
        positive_samples = torch.cat((sim_i_j, sim_j_i), dim=0).reshape(N, 1)
        negative_samples = sim[self.mask].reshape(N, -1)
        labels = torch.zeros(N).to(positive_samples.device).long()
        logits = torch.cat((positive_samples, negative_samples), dim=1)
        loss = self.criterion(logits, labels)
        loss /= N
        return loss




class hs_ince(NT_Xent):
    def __init__(self, batch_size, temperature,iterations=None, lam=None, q=None):
        super(hs_ince,self).__init__(batch_size, temperature, iterations, lam, q)
    
    def forward(self, z):
        z = F.normalize(z, dim=1)
        P = self.gibs_kernel(z)
        P = P / P.sum(dim=1, keepdim=True)
        P_tgt = torch.cat([torch.arange(self.batch_size) for i in range(2)], dim=0)
        P_tgt = ((P_tgt.unsqueeze(0) == P_tgt.unsqueeze(1)).float()).to(z.device)
        mask = torch.eye(P_tgt.shape[0]).to(z.device)
        P_tgt = P_tgt - mask
        loss = F.kl_div(P.log(), P_tgt, reduction='batchmean')
        #print('loss',loss)
        return loss



class gca_ince(NT_Xent):
    def __init__(self, batch_size, temperature,iterations=10, lam=None, q=None):
        super(gca_ince,self).__init__(batch_size, temperature, iterations, lam, q)
   


    def forward(self,z,C=None):
        z = F.normalize(z, dim=1)
        u = z.new_ones((z.size(0),), requires_grad=False)
        v = z.new_ones((z.size(0),), requires_grad=False)
        r,c = u,v
        K = self.gibs_kernel(z)
        for _ in range(self.iterations):
            u = r/(K @ v)
            v = c/(K.T @ u ) 
        P = (torch.diag(u)) @ K @ (torch.diag(v))
        P_tgt = torch.cat([torch.arange(self.batch_size) for i in range(2)], dim=0)
        P_tgt = ((P_tgt.unsqueeze(0) == P_tgt.unsqueeze(1)).float()).to(z.device)
        mask = torch.eye(P_tgt.shape[0]).to(z.device)
        P_tgt = P_tgt - mask
        loss = F.kl_div(P.log(), P_tgt, reduction='batchmean')
        return loss

    def compute_cost(self, z):
        z_scores =  (z @ z.t()).clamp(min=1e-7)  # normalized cosine similarity scores
        Cost = 1 - z_scores
        C =Cost.max()  # Assume C is large enough, using max value in Cost matrix
        diags = C * torch.eye(Cost.shape[0]).to(Cost.device)
        diagsoff = (1 - torch.eye(Cost.shape[0])).to(Cost.device) * Cost
        Cost = diags + diagsoff
        # C_scale =(1-z_scores)/self.epsilon
        # C_scale= C_scale+  torch.eye(C_scale.size(0)).to(C_scale.device) * 1e5
        # print('C',C_scale,C_scale.min(),C_scale.max()) 
        return Cost


    def gibs_kernel(self, x):
        Cost = self.compute_cost(x)
        kernel = torch.exp(-Cost / self.temperature)
        return kernel





class rince(NT_Xent):
    def __init__(self, batch_size, temperature, iterations=10, lam=0.01, q=0.6):
        """
        RINCE loss for contrastive learning.

        Args:
            batch_size (int): Number of samples per batch.
            temperature (float): Temperature parameter for scaling similarities.
            iterations (int, optional): Number of Sinkhorn iterations. Defaults to 10.
            lam (float, optional): Lambda parameter for weighting. Defaults to 0.01.
            q (float, optional): Q parameter for generalized cross-entropy. Defaults to 0.6.
        """
        super(rince, self).__init__(batch_size, temperature, iterations, lam, q)
 

    def forward(self, z, C=None):
        z = F.normalize(z, dim=1)
        N = 2 * self.batch_size 
        sim = torch.exp(torch.mm(z, z.t()) / self.temperature)
        #print('similarity',sim.shape,sim)
        self.mask = self.mask_correlated_samples(self.batch_size)
        sim_i_j = torch.diag(sim, self.batch_size )
        sim_j_i = torch.diag(sim, -self.batch_size )
        # We have 2N samples, but with Distributed training every GPU gets N examples too, resulting in: 2xNxN
        pos = torch.cat((sim_i_j, sim_j_i), dim=0)#.reshape(N, 1)
        neg = torch.sum(sim[self.mask].reshape(N, -1), dim=1)
        #neg = torch.sum(sim * neg_mask, 1)
        neg = ((self.lam*(pos + neg))**self.q) / self.q
        pos = -(pos**self.q) / self.q
        loss = pos.mean() + neg.mean()
        return loss


class hs_rince(NT_Xent):
    def __init__(self, batch_size, temperature, iterations=10, lam=0.01, q=0.6):
        """
        RINCE loss for contrastive learning.

        Args:
            batch_size (int): Number of samples per batch.
            temperature (float): Temperature parameter for scaling similarities.
            iterations (int, optional): Number of Sinkhorn iterations. Defaults to 10.
            lam (float, optional): Lambda parameter for weighting. Defaults to 0.01.
            q (float, optional): Q parameter for generalized cross-entropy. Defaults to 0.6.
        """
        super(hs_rince, self).__init__(batch_size, temperature, iterations, lam, q)

    
    def forward(self,z,C=None):
        z=F.normalize(z,dim=1)
        batch_size=z.size(0)//2
        P_tgt = torch.cat([torch.arange(batch_size) for i in range(2)], dim=0)
        P_tgt = ((P_tgt.unsqueeze(0) == P_tgt.unsqueeze(1)).float()).to(z.device)
        mask = torch.eye(P_tgt.shape[0], dtype=torch.bool).to(z.device)
        P_tgt = P_tgt.masked_fill(mask, 0) 
        u = z.new_ones((z.size(0),), requires_grad=False) /z.size(0)
        v = z.new_ones((z.size(0),), requires_grad=False) /z.size(0)
        K = self.gibs_kernel(z)
        u1 = u/(K @ v)
        P = (torch.diag(u1)) @ K @ (torch.diag(v))
        logits=(-(P[P_tgt.bool()]/u1).pow(self.q)/self.q).mean()+((self.lam*(P_tgt/u1).sum(axis=1)).pow(self.q)/self.q).mean()
        return logits




class gca_rince(NT_Xent):
    def __init__(self, batch_size, temperature, iterations=10, lam=0.01, q=0.6):
        super(gca_rince, self).__init__(batch_size,temperature,iterations=iterations, lam=lam, q=q)


    def forward(self,z):
        N = 2 * self.batch_size 
        z = F.normalize(z, dim=1)
        batch_size=z.size(0)//2

        P_tgt = torch.cat([torch.arange(batch_size) for i in range(2)], dim=0)
        P_tgt = ((P_tgt.unsqueeze(0) == P_tgt.unsqueeze(1)).long()).to(z.device)
        mask = torch.eye(z.size(0), dtype=torch.bool, device=z.device)
        P_tgt = P_tgt.masked_fill(mask, 0) 
        u = z.new_ones((z.size(0),), requires_grad=False) /z.size(0)
        v = z.new_ones((z.size(0),), requires_grad=False) /z.size(0)
        P = self.gibs_kernel(z)
        for _ in range(self.iterations):
            P=P/(P.sum(axis=1,keepdim=True))
            P=P/(P.sum(axis=0,keepdim=True)).T
        sim_i_j = torch.diag(P, self.batch_size )
        sim_j_i = torch.diag(P, -self.batch_size )
        # We have 2N samples, but with Distributed training every GPU gets N examples too, resulting in: 2xNxN
        pos = torch.cat((sim_i_j, sim_j_i), dim=0).reshape(N, 1)
        neg = P[self.mask].reshape(N, -1)

        neg = ((self.lam*(pos + neg))**self.q) / self.q
        pos = -(pos**self.q) / self.q
        loss = pos.mean() + neg.mean()
        #logits=(-(P[P_tgt.bool()]/u).pow(self.q)/self.q).mean()+((self.lam*(P_tgt/u).sum(axis=1)).pow(self.q)/self.q).mean()
        #logits=(-(P[P_tgt.bool()]/u).pow(self.q)/self.q)+((self.lam*(P_tgt/u).sum(axis=1)).pow(self.q)/self.q)
        return loss


class gca_uot(NT_Xent):
    def __init__(self, batch_size, temperature, iterations=10, lam=0.01, q=0.6,relax_items1=1e-4, relax_items2=1e-4,r1=1.0, r2=1.0):
        super(gca_uot,self).__init__(batch_size,temperature,iterations=iterations, lam=lam, q=q)
        self.tau=1e5 #1e5
        self.stopThr=1e-16
        self.relax_items1=relax_items1
        self.relax_items2=relax_items2
        self.lam=lam
        self.q=q
        self.epsilon=temperature
        self.iterations=10
        self.relax_items1=relax_items1
        self.relax_items2=relax_items2
        self.r1 = r1
        self.r2 = r2
    
    def forward(self,z):
        z = F.normalize(z, dim=1)
        batch_size=z.size(0)//2
        P_tgt = torch.cat([torch.arange(batch_size) for i in range(2)], dim=0)
        P_tgt = ((P_tgt.unsqueeze(0) == P_tgt.unsqueeze(1)).float()).to(z.device)
        mask = torch.eye(P_tgt.shape[0]).to(z.device)
        P_tgt =P_tgt - mask
        P = self.gibs_kernel(z)
        # Initialize cost, relaxation exponents, and dual variables for OT
        #C = self.compute_cost(z)
        f1 = self.relax_items1 / (self.relax_items1 + self.epsilon)
        f2 = self.relax_items2 / (self.relax_items2 + self.epsilon)
        # f = torch.zeros(P.size(0), device=P.device, dtype=P.dtype, requires_grad=False)
        # g = torch.zeros(P.size(1), device=P.device, dtype=P.dtype, requires_grad=False)

        # Sinkhorn-like iterations with relaxation
        for i in range(self.iterations):
            P_prev = P.clone()            
            row_sum = P.sum(dim=1, keepdim=True)
            P =  P * (1.0 / row_sum) ** f1   # relaxation factor
            col_sum = P.sum(dim=0, keepdim=True).T
            # Compute column sums and normalize columns with relaxation
            P  =P * (1.0 / col_sum)  ** f2 

            # if torch.any(P > self.tau):
            #     f += self.epsilon * torch.log(torch.max(P, dim=1)[0])
            #     g += self.epsilon * torch.log(torch.max(P, dim=0)[0])
            #     P = torch.exp((f[:, None] + g[None, :] - C) / self.epsilon).clamp(min=self.stopThr)

            # Check for convergence or numerical issues
            if torch.any(torch.isnan(P)) or torch.any(torch.isinf(P)):
                P = P_prev
                break
            # P=P/(P.sum(axis=1,keepdim=True))
            # P =(P/P.sum(axis=0,keepdim=True)).T
        kl_logits = F.kl_div(P.log(), P_tgt, reduction='batchmean')
        u1 = P.sum(axis=1)
        sim_i_j = torch.diag(P, self.batch_size )
        sim_j_i = torch.diag(P, -self.batch_size)
        # We have 2N samples, but with Distributed training every GPU gets N examples too, resulting in: 2xNxN
        pos = torch.cat((sim_i_j, sim_j_i), dim=0).reshape(2*batch_size, 1)
        neg = P[self.mask].reshape(2*batch_size, -1)
        neg = torch.sum(P[self.mask].reshape(2*batch_size, -1), dim=1)
        neg = ((self.lam*(pos + neg))**self.q) / self.q
        pos = -(pos**self.q) / self.q
        w1_logits = pos.mean() + neg.mean()
        #w1_logits=-((P[P_tgt.bool()]/u1)**self.q/self.q).mean()+((self.lam*(P_tgt/u1).sum(axis=1))**self.q/self.q).mean()        
        #print('kl_logits',kl_logits,'w1_logits',w1_logits)
        loss = self.r1*w1_logits + self.r2*kl_logits
        #print('loss',loss,w1_logits,kl_logits)
        return loss



    
class iot(torch.nn.Module):
    def __init__(self,  batch_size, temperature, iterations=10, lambda_p=0.2, q=None):
        super().__init__()
        self.epsilon = temperature
        self.iterations = iterations
        self.lambda_p = lambda_p
    def forward(self, z):
        # Concatenate z and z' (Cat({zi}, {z'j})) and calculate cosine similarity
        z = F.normalize(z, dim=1)
        cos_sim = F.cosine_similarity(z.unsqueeze(1), z.unsqueeze(0), dim=2)
        # Compute Cost matrix
        Cost = 1 - cos_sim
        Sim = Cost.shape[0]
        C = Cost.max()  # Assume C is large enough, using max value in Cost matrix
        diags = C * torch.eye(Sim).to(Cost.device)
        diagsoff = (1 - torch.eye(Sim)).to(Cost.device) * Cost
        Cost = diags + diagsoff
        # Initialize Pθ
        Probs = torch.exp(-Cost / self.epsilon)
        # Sinkhorn iterations for normalization
        for _ in range(self.iterations):
            # Row normalization
            Probs = Probs / (Probs.sum(dim=1, keepdim=True))
            # Column normalization
            Probs = (Probs / (Probs.sum(dim=0, keepdim=True) )).T
        # Selecting the positive pairs
        N = z.shape[0] // 2
        Probi = Probs.diag(N)
        Probi = Probi.unsqueeze(0)
        Probj = Probs.diag(-N)
        Probj = Probj.unsqueeze(0)
        Prob_plus = torch.cat((Probi, Probj), dim=0)
        # Compute contrastive loss (L_IOT-CL)
        L_IOT_CL = -torch.log(Prob_plus).sum() / (2 * N)
        return L_IOT_CL


    
class iot_uni(torch.nn.Module):
    def __init__(self,  batch_size, temperature, iterations=10, lambda_p=0.2, q=None):
        super().__init__()
        self.epsilon = temperature
        self.iterations = iterations
        self.lambda_p = lambda_p
    def forward(self, z):
        # Concatenate z and z' (Cat({zi}, {z'j})) and calculate cosine similarity
        z = F.normalize(z, dim=1)
        cos_sim = F.cosine_similarity(z.unsqueeze(1), z.unsqueeze(0), dim=2)
        # Compute Cost matrix
        Cost = 1 - cos_sim
        Sim = Cost.shape[0]
        C = Cost.max()  # Assume C is large enough, using max value in Cost matrix
        diags = C * torch.eye(Sim).to(Cost.device)
        diagsoff = (1 - torch.eye(Sim)).to(Cost.device) * Cost
        Cost = diags + diagsoff
        # Initialize Pθ
        Probs = torch.exp(-Cost / self.epsilon)
        # Sinkhorn iterations for normalization
        for _ in range(self.iterations):
            # Row normalization
            Probs = Probs / (Probs.sum(dim=1, keepdim=True))
            # Column normalization
            Probs = (Probs / (Probs.sum(dim=0, keepdim=True) )).T
        # Selecting the positive pairs
        N = z.shape[0] // 2
        Probi = Probs.diag(N)
        Probi = Probi.unsqueeze(0)
        Probj = Probs.diag(-N)
        Probj = Probj.unsqueeze(0)
        Prob_plus = torch.cat((Probi, Probj), dim=0)
        # Compute contrastive loss (L_IOT-CL)
        L_IOT_CL = -torch.log(Prob_plus).sum() / (2 * N)
        # # Compute uniformity loss component (Q^θ)
        Q_theta = Probs.clone()
        mean_off_diagonal = Probs.masked_select(~torch.eye(Sim, dtype=bool).to(Cost.device)).mean()
        Q_theta[~torch.eye(Sim, dtype=bool).to(Cost.device)] = mean_off_diagonal
        # Compute KL divergence for uniformity loss
        uniformity_loss = F.kl_div(Probs.log(), Q_theta, reduction='batchmean')
        Loss = L_IOT_CL + self.lambda_p * uniformity_loss
        return Loss









