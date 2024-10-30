import numpy as np
import torch

class TruncNormal(torch.nn.Module):
    def __init__(
        self,
        n_dims:int, 
        fix_std:float=None,
    ) -> None:
        
        torch.nn.Module.__init__(self)

        self.min_val = 0.0
        self.max_val = 1.0
        self.min_log_std = -4.0
        self.max_log_std = 2.0
        self.fix_std = fix_std
        self.pre_mean = torch.nn.Parameter(torch.zeros(n_dims))
        if self.fix_std is not None:
            self.pre_std = torch.nn.Parameter(torch.zeros(n_dims))

    ################
    # Public Methods
    ################

    def getMeanStd(self):
        mean = torch.sigmoid(self.pre_mean)*(self.max_val - self.min_val) + self.min_val
        if self.fix_std:
            std = torch.ones_like(mean)*self.fix_std
        else:
            std = torch.exp(torch.clamp(self.pre_std, self.min_log_std, self.max_log_std))
        return mean, std

    def getMeanLogStd(self):
        mean = torch.sigmoid(self.pre_mean)*(self.max_val - self.min_val) + self.min_val
        if self.fix_std:
            log_std = torch.log(torch.ones_like(mean)*self.fix_std)
        else:
            log_std = torch.clamp(self.pre_std, self.min_log_std, self.max_log_std)
        return mean, log_std

    def sample(self):
        mean, std = self.getMeanStd()
        alpha = (self.min_val - mean)/std
        beta = (self.max_val - mean)/std
        z = self._Phi(beta) - self._Phi(alpha)
        u = torch.rand_like(mean)
        sampled_value = z*u + self._Phi(alpha)
        sampled_value = torch.erfinv(2.0*sampled_value - 1.0)
        sampled_value = np.sqrt(2.0)*std*sampled_value + mean
        return sampled_value

    def samples(self, batch_size):
        mean, std = self.getMeanStd()
        means = mean.unsqueeze(0).expand(batch_size, -1)
        stds = std.unsqueeze(0).expand(batch_size, -1)
        alpha = (self.min_val - means)/stds # (batch_size, n_dims)
        beta = (self.max_val - means)/stds # (batch_size, n_dims)
        z = self._Phi(beta) - self._Phi(alpha) # (batch_size, n_dims)
        u = torch.rand_like(means) # (batch_size, n_dims)
        sampled_value = z*u + self._Phi(alpha)
        sampled_value = torch.erfinv(2.0*sampled_value - 1.0)
        sampled_value = np.sqrt(2.0)*stds*sampled_value + means
        return sampled_value

    def getProb(self, x):
        mean, std = self.getMeanStd()
        alpha = (self.min_val - mean)/std
        beta = (self.max_val - mean)/std
        z = self._Phi(beta) - self._Phi(alpha)
        mask = (x >= self.min_val) & (x <= self.max_val)
        pdf = self._phi((x - mean)/std)/(std*z)
        return mask*pdf

    def getLogProbs(self, x):
        batch_size = x.shape[0]
        mean, log_std = self.getMeanLogStd()
        means = mean.unsqueeze(0).expand(batch_size, -1)
        log_stds = log_std.unsqueeze(0).expand(batch_size, -1)
        stds = torch.exp(log_stds)

        alpha = (self.min_val - means)/stds
        beta = (self.max_val - means)/stds
        z = self._Phi(beta) - self._Phi(alpha)

        zeta = (x - means)/stds
        log_phi = -0.5*(zeta**2) - np.log(np.sqrt(2.0*np.pi))
        log_pdf = log_phi - log_stds - torch.log(z)
        return log_pdf

    def getEntropy(self):
        mean, log_std = self.getMeanLogStd()
        std = torch.exp(log_std)
        alpha = (self.min_val - mean)/std
        beta = (self.max_val - mean)/std
        z = self._Phi(beta) - self._Phi(alpha)
        return (np.log(np.sqrt(2.0*np.pi*np.e)) + log_std + torch.log(z) \
                + (alpha*self._phi(alpha) - beta*self._phi(beta))/(2.0*z)).mean()

    #################
    # Private Methods
    #################

    def _Phi(self, x):
        return 0.5*(1.0 + torch.erf(x/np.sqrt(2.0)))
    
    def _phi(self, x):
        return torch.exp(-0.5*(x**2))/np.sqrt(2.0*np.pi)