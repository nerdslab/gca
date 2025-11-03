import torch
import torch.nn.functional as F


class NT_Xent(torch.nn.Module):
    def __init__(self,  batch_size, epsilon, iterations=10, lam=None, q=None):
        """
        Initializes the NT_Xent loss module.
        
        Args:
            temperature (float): Temperature parameter for scaling similarities.
        """
        super(NT_Xent, self).__init__()
        self.temperature = epsilon

    def forward(self, z):
        """
        Compute the NT-Xent loss.

        Args:
            z (torch.Tensor): Input tensor of shape (2N, D), where N is the batch size
                              and D is the dimensionality of the embeddings.

        Returns:
            torch.Tensor: Scalar loss value.
        """
        # Normalize embeddings
        z = F.normalize(z, dim=1)
        
        # Compute pairwise cosine similarities
        z_scores = (z @ z.t()).clamp(min=1e-7)  # Avoid numerical instability
        
        # Scale similarities with temperature
        z_scale = z_scores / self.temperature
        
        # Mask self-similarities with a large negative value
        z_scale = z_scale - torch.eye(z_scale.size(0)).to(z_scale.device) * 1e5

        # Define targets for contrastive learning
        targets = torch.arange(z.size(0))
        targets[::2] += 1  # For even indices, target is the next odd index
        targets[1::2] -= 1  # For odd indices, target is the previous even index

        # Compute cross-entropy loss
        loss = F.cross_entropy(z_scale, targets.long().to(z_scale.device))
        return loss


class base(torch.nn.Module):
    def  __init__(self, batch_size, epsilon, iterations=10, lam=None, q=None):
        super(base,self).__init__()
        self.batch_size = batch_size
        self.epsilon = epsilon
        self.iterations = iterations
        self.lam = lam
        self.q = q
    
    def compute_cost(self, z):
        z =F.normalize(z, dim=1)
        z_scores = (z @ z.t()).clamp(min=1e-7) 
        Cost = 1 - z_scores
        C = Cost.max()  # Assume C is large enough, using max value in Cost matrix
        diags = C * torch.eye(Cost.shape[0]).to(Cost.device)
        diagsoff = (1 - torch.eye(Cost.shape[0])).to(Cost.device) * Cost
        Cost = diags + diagsoff
        return Cost
    
    def gibs_kernel(self, x):
        Cost = self.compute_cost(x)
        # C_scale =(1-sim_scores)/self.epsilon
        # C= C_scale+  torch.eye(C_scale.size(0)).to(C_scale.device) * 1e5
        kernel = torch.exp(-Cost / self.epsilon)
        return kernel
    
    def forward(self,z,C=None):
        raise NotImplementedError

class gca_ince(base):
    def  __init__(self, batch_size, epsilon, iterations=10, lam=None, q=None):
        super(gca_ince,self).__init__(batch_size, epsilon, iterations=iterations, lam=lam, q=q)

    def forward(self,z,C=None):
        batch_size=z.size(0)//2
        P_tgt = torch.cat([torch.arange(batch_size) for i in range(2)], dim=0)
        P_tgt = ((P_tgt.unsqueeze(0) == P_tgt.unsqueeze(1)).float()).to(z.device)
        mask = torch.eye(P_tgt.shape[0]).to(z.device)
        P_tgt = P_tgt - mask
        u = z.new_ones((z.size(0),), requires_grad=False)
        v = z.new_ones((z.size(0),), requires_grad=False)
        r,c = u,v
        K = self.gibs_kernel(z)
        for _ in range(self.iterations):
            u = r/(K @ v)
            v = c/(K.T @ u ) 
        P = (torch.diag(u)) @ K @ (torch.diag(v))
        targets = torch.arange(z.size()[0])
        targets[::2] += 1  # target of 2k element is 2k+1
        targets[1::2] -= 1  # target of 2k+1 element is 2k
        # Create a mask for the diagonal elements
        logits=F.cross_entropy(P.log(), targets.long().to(P.device))
        return logits.sum()#/z.size(0)

class hs_ince(base):
    def  __init__(self, batch_size, epsilon, iterations=10, lam=None, q=None):
        super(hs_ince,self).__init__(batch_size, epsilon, iterations=iterations, lam=lam, q=q)

    
    def forward(self,z,C=None):
        batch_size=z.size(0)//2
        P_tgt = torch.cat([torch.arange(batch_size) for i in range(2)], dim=0)
        P_tgt = ((P_tgt.unsqueeze(0) == P_tgt.unsqueeze(1)).float()).to(z.device)
        mask = torch.eye(P_tgt.shape[0]).to(z.device)
        P_tgt = P_tgt - mask
        u = z.new_ones((z.size(0),), requires_grad=False)
        v = z.new_ones((z.size(0),), requires_grad=False)
        K = self.gibs_kernel(z)
        u1 = u/(K @ v)
        P = (torch.diag(u1)) @ K @ (torch.diag(v))
        # logits = F.cross_entropy(torch.log(P), targets.long())
        #print('logP',logP,logP.min(),logP.max())
        targets = torch.arange(z.size()[0])
        targets[::2] += 1  # target of 2k element is 2k+1
        targets[1::2] -= 1  # target of 2k+1 element is 2k
        # Create a mask for the diagonal elements
        logits=F.cross_entropy(P.log(), targets.long().to(P.device))
        #logits = I*torch.diag(torch.div(I,P)).log()#/x.size(0)
        return logits.sum()#/z.size(0)



class rince(base):
    def  __init__(self, batch_size, epsilon, iterations=10, lam=None, q=None):
        super(rince, self).__init__(batch_size, epsilon, iterations=iterations, lam=lam, q=q)
        self.lam = lam
        self.q = q
        self.epsilon = epsilon
        self.batch_size=512
        self.world_size=1
        self.num_pos=2
        self.mask = self.mask_correlated_samples(self.batch_size, 1)
        print('q is',q,'lam is',lam,'epsilon is',epsilon)

    def mask_correlated_samples(self, batch_size, world_size):
        N = 2 * batch_size * world_size
        mask = torch.ones((N, N), dtype=bool)
        mask = mask.fill_diagonal_(0)
        for i in range(batch_size * world_size):
            mask[i, batch_size * world_size + i] = 0
            mask[batch_size * world_size + i, i] = 0
        return mask

    def forward(self, z):
        z = F.normalize(z, dim=1)
        z_scores =  (z @ z.t()).clamp(min=1e-7)  # normalized cosine similarity scores
        z_scale = z_scores / self.epsilon   # scale with temperature
        logits = z_scale - torch.eye(z_scale.size(0)).to(z_scale.device) * z_scale.max()
        targets = torch.arange(z.size()[0])
        targets[::2] += 1  # target of 2k element is 2k+1
        targets[1::2] -= 1  # target of 2k+1 element is 2k
        targets = targets.to(z.device)
        
        num_classes = logits.size(1)
        class_weights = torch.ones(num_classes).to(logits.device)
        class_weights *= self.lam  # Set all weights to lambda
        class_weights[targets.unique()] = 1.0  # Set weights of positive classes to 1
        loss = F.cross_entropy(logits, targets, weight=class_weights)
        adjusted_loss = (loss ** self.q) / self.q
        return adjusted_loss


    def generalized_cross_entropy(self, logits, targets, q, lam):
        # Compute probabilities
        probs = F.softmax(logits, dim=1)
        # Apply lambda and q
        probs_q = (lam * probs) ** q
        # One-hot encode the targets
        targets_one_hot = F.one_hot(targets, num_classes=probs.size(1)).float()
        # Compute the generalized cross-entropy loss
        loss = torch.sum(-targets_one_hot * probs_q / q, dim=1)
        loss = loss.mean()
        return loss
    

class hs_rince(base):
    def  __init__(self, batch_size, epsilon, iterations=10, lam=None, q=None):
        super(hs_rince, self).__init__(batch_size, epsilon, iterations=iterations, lam=lam, q=q)
        self.lam = lam
        self.q = q
    
    def forward(self,z,C=None):
        # batch_size=z.size(0)//2
        # P_tgt = torch.cat([torch.arange(batch_size) for i in range(2)], dim=0)
        # P_tgt = ((P_tgt.unsqueeze(0) == P_tgt.unsqueeze(1)).float()).to(z.device)
        # mask = torch.eye(P_tgt.shape[0]).to(z.device)
        # P_tgt = (P_tgt - mask)/z.size(0)
        ids = torch.arange(z.size(0), device=z.device) // 2   # [0,0,1,1,2,2,...]
        P_tgt = (ids[:, None] == ids[None, :]).float()
        P_tgt.fill_diagonal_(0)  
        u = z.new_ones((z.size(0),), requires_grad=False) /z.size(0)
        v = z.new_ones((z.size(0),), requires_grad=False) /z.size(0)
        K = self.gibs_kernel(z)
        # Normalize to make them probability distributions
        u1 = u/(K @ v)
        P = (torch.diag(u1)) @ K @ (torch.diag(v))
        logits=-(P[P_tgt.bool()]/u1).pow(self.q)/self.q+(self.lam*(P_tgt/u1).sum(axis=1)).pow(self.q)/self.q
        return logits.sum()/z.size(0)




class gca_rince(base):
    def  __init__(self, batch_size, epsilon, iterations=10, lam=None, q=None):
        super(gca_rince,self).__init__(batch_size, epsilon, iterations=iterations, lam=lam, q=q)
        self.lam=lam
        self.q=q
        self.iterations=10
        self.mask = self.mask_correlated_samples(512, 1)

    def forward(self,z,C=None):
        z = F.normalize(z, dim=1)
        # batch_size=z.size(0)//2
        # P_tgt = torch.cat([torch.arange(batch_size) for i in range(2)], dim=0)
        # P_tgt = ((P_tgt.unsqueeze(0) == P_tgt.unsqueeze(1)).long()).to(z.device)
        # mask = torch.eye(z.size(0), dtype=torch.bool, device=z.device)
        # P_tgt = P_tgt.masked_fill(mask, 0) 
        u = z.new_ones((z.size(0),), requires_grad=False) /z.size(0)
        v = z.new_ones((z.size(0),), requires_grad=False) /z.size(0)
        K = self.gibs_kernel(z)
        a,b=u,v
        for _ in range(self.iterations):
            u = a/(K @ v)
            v = b/(K.T @ u)
        P=(torch.diag(u)) @ K @ (torch.diag(v))
        targets = torch.arange(z.size()[0])
        targets[::2] += 1  # target of 2k element is 2k+1
        targets[1::2] -= 1  # target of 2k+1 element is 2k
        targets = targets.to(z.device)
        logits= P.log()
        num_classes = logits.size(1)
        class_weights = torch.ones(num_classes).to(logits.device)
        class_weights *= self.lam  # Set all weights to lambda
        class_weights[targets.unique()] = 1.0  # Set weights of positive classes to 1
        loss = F.cross_entropy(logits, targets, weight=class_weights)
        adjusted_loss = (loss ** self.q) / self.q
        return adjusted_loss

    def mask_correlated_samples(self, batch_size, world_size):
        N = 2 * batch_size * world_size
        mask = torch.ones((N, N), dtype=bool)
        mask = mask.fill_diagonal_(0)
        for i in range(batch_size * world_size):
            mask[i, batch_size * world_size + i] = 0
            mask[batch_size * world_size + i, i] = 0
        return mask



class gca_uot(base):
    def __init__(self, batch_size, epsilon, iterations=10, lam=0.01, q=0.6,relax_items1=1e-4, relax_items2=1e-4,r1=1.0, r2=1.0):
        super(gca_uot,self).__init__(batch_size, epsilon, iterations=iterations, lam=lam, q=q)
        self.tau=1e5 #1e5
        self.stopThr=1e-16
        self.relax_items1=relax_items1
        self.relax_items2=relax_items2
        self.lam=lam
        self.q=q
        self.r1=r1
        self.r2=r2
    
    def forward(self,z,C=None):
        # batch_size=z.size(0)//2
        z=F.normalize(z,dim=1)
        # P_tgt = torch.cat([torch.arange(batch_size) for i in range(2)], dim=0)
        # P_tgt = ((P_tgt.unsqueeze(0) == P_tgt.unsqueeze(1)).float()).to(z.device)
        # mask = torch.eye(P_tgt.shape[0]).to(z.device)
        # P_tgt = (P_tgt - mask)/z.size(0)
        ids = torch.arange(z.size(0), device=z.device) // 2   # [0,0,1,1,2,2,...]
        P_tgt = (ids[:, None] == ids[None, :]).float()
        P_tgt.fill_diagonal_(0)  

        u = z.new_ones((z.size(0),), requires_grad=False) /z.size(0)
        v = z.new_ones((z.size(0),), requires_grad=False) /z.size(0)
        K = self.gibs_kernel(z)
        a,b=u,v
        f1 = self.relax_items1 / (self.relax_items1 + self.epsilon)
        f2 = self.relax_items2 / (self.relax_items2+ self.epsilon)
        f = torch.zeros_like(u, requires_grad=False)
        g = torch.zeros_like(v, requires_grad=False)
        C = self.compute_cost(z) if C is None else C
        for i in range(self.iterations):
            uprev = u
            vprev = v
            f_ = torch.exp(- f / (self.epsilon + self.relax_items1))
            g_ = torch.exp(- g / (self.epsilon + self.relax_items2))
            u = ((a / (K@v + 1e-16)) ** f1) * f_
            v = ((b / (K.T@u + 1e-16)) ** f2) * g_
            if torch.any(u > self.tau) or torch.any(v > self.tau):
                f = f + self.epsilon * torch.log(torch.max(u))
                g = g + self.epsilon * torch.log(torch.max(v))
                K = torch.exp((f[:, None] + g[None, :] - C) / self.epsilon)
                v = torch.ones(v.shape, type_as=v)
            if (torch.any(K.T@u == 0.) or torch.any(torch.isnan(u)) or torch.any(torch.isnan(v))
                    or torch.any(torch.isinf(u)) or torch.any(torch.isinf(v))):
                #warnings.warn('We have reached the machine precision %s' % i)
                u = uprev
                v = vprev
                break
        logu = f / self.epsilon + torch.log(u)
        logv = g / self.epsilon + torch.log(v)
        P = torch.exp(logu[:, None] + logv[None, :] - C / self.epsilon)
        # Selecting the positive pairs
        targets = torch.arange(z.size()[0])
        targets[::2] += 1  # target of 2k element is 2k+1
        targets[1::2] -= 1  # target of 2k+1 element is 2k
        kl_logits=F.cross_entropy(P.log(), targets.long().to(P.device))
        logits=self.r2*(-(P[P_tgt.bool()]/u).pow(self.q)/self.q+(self.lam*(P_tgt/u).sum(axis=1)).pow(self.q)/self.q)+self.r1*kl_logits
        return logits.mean()


class iot_uni(base):
    def  __init__(self, batch_size, epsilon, iterations=10, lam=None, q=None,lambda_p=0.2):
        super(iot_uni,self).__init__(batch_size, epsilon, iterations=iterations, lam=lam, q=q)
        self.lambda_p = lambda_p

    def gibs_kernel(self, x, y):
        C = self.compute_cost(x, y)
        return torch.exp(-C / self.epsilon)

    def compute_cost(self, x, y):
        cos_sim = F.cosine_similarity(x.unsqueeze(1), y.unsqueeze(0), dim=2)
        Cost = 1 - cos_sim
        return Cost

    def forward(self,z, C=None):
        # Concatenate z and z' (Cat({zi}, {z'j})) and calculate cosine similarity
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
        # Compute uniformity loss component (Q^θ)
        N = z.shape[0] // 2
        Probi = Probs.diag(N)
        Probi = Probi.unsqueeze(0)
        Probj = Probs.diag(-N)
        Probj = Probj.unsqueeze(0)
        Prob_plus = torch.cat((Probi, Probj), dim=0)
        # Compute contrastive loss (L_IOT-CL)
        L_IOT_CL = -torch.log(Prob_plus ).sum() / (2 * N)
        Q_theta = Probs.clone()
        mean_off_diagonal = Probs.masked_select(~torch.eye(Probs.size(0), dtype=bool).to(Probs.device)).mean()
        Q_theta[~torch.eye(Probs.size(0), dtype=bool).to(Probs.device)] = mean_off_diagonal
        # Compute KL divergence for uniformity loss
        uniformity_loss = F.kl_div(Probs.log(), Q_theta)
        # Final loss
        Loss = L_IOT_CL + self.lambda_p * uniformity_loss
        return Loss


class iot(base):
    def  __init__(self, batch_size, epsilon, iterations=10, lam=None, q=None):
        super(iot,self).__init__(batch_size, epsilon, iterations=iterations, lam=lam, q=q)

    def forward(self, z):
        # Concatenate z and z' (Cat({zi}, {z'j})) and calculate cosine similarity
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
        # Compute Loss
        Loss = -torch.log(Prob_plus ).sum() / (2 * N)
        return Loss 
