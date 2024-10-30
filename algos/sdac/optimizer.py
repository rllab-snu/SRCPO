from algos.common.optimizer_base import TROptimizer, flatGrad

from qpsolvers import solve_qp
from scipy import optimize
from copy import deepcopy
import numpy as np
import torch

EPS = 1e-8

class SDACOptimizer(TROptimizer):
    def __init__(
        self, device,
        actor, 
        damping_coeff, 
        num_conjugate, 
        line_decay, 
        max_kl, 
        con_thresholds, 
        con_zeta,
        con_entropy) -> None:

        super().__init__(device, actor, damping_coeff, num_conjugate, line_decay, max_kl)
        self.con_thresholds = con_thresholds
        self.con_zeta = con_zeta
        self.con_entropy = con_entropy
        self.n_cons = len(self.con_thresholds)

        # for solver
        self.bounds = optimize.Bounds(np.zeros(self.n_cons + 1), np.ones(self.n_cons + 1)*np.inf)
        def dual(x, g_H_inv_g, r_vector, S_mat, c_vector, max_kl):
            lam_vector = x[:-1]
            nu_scalar = x[-1]
            objective = (g_H_inv_g - 2.0*np.dot(r_vector, lam_vector) + np.dot(lam_vector, S_mat@lam_vector))/(2.0*nu_scalar + EPS) \
                            - np.dot(lam_vector, c_vector) + nu_scalar*max_kl
            return objective
        def dualJac(x, g_H_inv_g, r_vector, S_mat, c_vector, max_kl):
            lam_vector = x[:-1]
            nu_scalar = x[-1]
            jacobian = np.zeros_like(x)
            jacobian[:-1] = (S_mat@lam_vector - r_vector)/(nu_scalar + EPS) - c_vector
            jacobian[-1] = max_kl - (g_H_inv_g - 2.0*np.dot(r_vector, lam_vector) + np.dot(lam_vector, S_mat@lam_vector))/(2.0*(nu_scalar**2) + EPS)
            return jacobian
        self.dual = dual
        self.dualJac = dualJac


    def step(self, get_obj_con_kl, mu_kl=0.0):
        # for adaptive kl
        max_kl = self._getMaxKL(mu_kl)

        # calculate objective, constraints, kl
        objective, constraints, kl = get_obj_con_kl()
        self._computeKLGrad(kl)

        # for objective
        g_tensor = flatGrad(objective, self.actor.parameters(), retain_graph=True)
        H_inv_g_tensor = self._conjugateGradient(g_tensor)
        approx_g_tensor = self._Hx(H_inv_g_tensor)
        g_H_inv_g_tensor = torch.dot(approx_g_tensor, H_inv_g_tensor)

        # for constraints
        con_vals = []
        b_tensors = []
        H_inv_b_tensors = []
        c_scalars = []
        max_c_scalars = []
        safety_mode = False
        for con_idx in range(self.n_cons):
            con_val = constraints[con_idx].item()
            b_tensor = flatGrad(constraints[con_idx], self.actor.parameters(), retain_graph=True)
            H_inv_b_tensor = self._conjugateGradient(b_tensor)
            approx_b_tensor = self._Hx(H_inv_b_tensor)
            b_H_inv_b_tensor = torch.dot(approx_b_tensor, H_inv_b_tensor)
            max_c_scalar = np.sqrt(np.clip(2.0*max_kl*b_H_inv_b_tensor.item(), 0.0, np.inf))
            c_scalar = min(max_c_scalar, con_val - self.con_thresholds[con_idx])
            if con_val > self.con_thresholds[con_idx]:
                safety_mode = True
            con_vals.append(con_val)
            b_tensors.append(approx_b_tensor)
            H_inv_b_tensors.append(H_inv_b_tensor)
            c_scalars.append(c_scalar)
            max_c_scalars.append(max_c_scalar)

        with torch.no_grad():
            # convert to tensor
            B_tensor = torch.stack(b_tensors).T
            H_inv_B_tensor = torch.stack(H_inv_b_tensors).T
            S_tensor = B_tensor.T@H_inv_B_tensor
            r_tensor = approx_g_tensor@H_inv_B_tensor

            # convert to numpy
            S_mat = S_tensor.detach().cpu().numpy().astype(np.float64)
            S_mat = 0.5*(S_mat + S_mat.T)
            # ===== to ensure S_mat is invertible ===== #
            min_eig_val = min(0.0, np.min(np.linalg.eigvals(S_mat)))
            S_mat += np.eye(S_mat.shape[0])*(-min_eig_val + EPS)
            # ========================================= #
            r_vector = r_tensor.detach().cpu().numpy()
            g_H_inv_g_scalar = g_H_inv_g_tensor.detach().cpu().numpy()
            c_vector = np.array(c_scalars)

            # find scaling factor
            const_lam_vector = solve_qp(P=S_mat, q=-c_vector, lb=np.zeros_like(c_vector))
            approx_kl = 0.5*np.dot(const_lam_vector, S_mat@const_lam_vector)
            scaling = 1.0 if approx_kl <= max_kl else np.sqrt(max_kl/approx_kl)

            # find search direction
            if approx_kl/max_kl - 1.0 > -0.001:
                for c_idx in range(len(c_vector)):
                    c_vector[c_idx] = min(max_c_scalars[c_idx], con_vals[c_idx] - self.con_thresholds[c_idx] + self.con_zeta)
                const_lam_vector = solve_qp(P=S_mat, q=-c_vector, lb=np.zeros_like(c_vector))
                approx_kl = 0.5*np.dot(const_lam_vector, S_mat@const_lam_vector)
                scaling = np.sqrt(max_kl/approx_kl)
                lam_tensor = torch.tensor(const_lam_vector, device=self.device, dtype=torch.float32)
                delta_theta = -scaling*H_inv_B_tensor@lam_tensor
            else:
                x0 = np.ones(self.n_cons + 1)
                res = optimize.minimize(\
                    self.dual, x0, method='trust-constr', jac=self.dualJac,
                    args=(g_H_inv_g_scalar, r_vector, S_mat, c_vector, max_kl), 
                    bounds=self.bounds, options={'disp': True, 'maxiter': 200}
                )
                if res.success:
                    lam_vector, nu_scalar = res.x[:-1], res.x[-1]
                    lam_tensor = torch.tensor(lam_vector, device=self.device, dtype=torch.float32)
                    delta_theta = (H_inv_g_tensor - H_inv_B_tensor@lam_tensor)/(nu_scalar + EPS)
                else:
                    # there is no solution -> only minimize costs.
                    lam_tensor = torch.tensor(const_lam_vector, device=self.device, dtype=torch.float32)
                    delta_theta = -scaling*H_inv_B_tensor@lam_tensor

            # update
            init_theta = torch.cat([t.view(-1) for t in self.actor.parameters()]).clone().detach()
            self._applyParams(delta_theta + init_theta)


        objective, constraints, kl = get_obj_con_kl()
        return objective, constraints, kl, safety_mode
