from algos.common.optimizer_base import TROptimizer, flatGrad

from qpsolvers import solve_qp
import numpy as np
import torch

EPS = 1e-8


class MultiTROptimizer(TROptimizer):
    def __init__(
        self, 
        device,
        actor, 
        damping_coeff, 
        num_conjugate, 
        line_decay, 
        max_kl, 
        n_objs,
        con_thresholds, 
        con_entropy,
    ) -> None:

        super().__init__(
            device, actor, damping_coeff, num_conjugate, 
            line_decay, max_kl)

        self.con_thresholds = con_thresholds
        self.con_entropy = con_entropy
        self.n_objs = n_objs
        self.n_cons = len(self.con_thresholds)

        # inner variables
        self.B_tensor = torch.zeros(
            (self.n_objs + self.n_cons, self.n_params), 
            device=self.device, dtype=torch.float32)
        self.H_inv_B_tensor = torch.zeros(
            (self.n_objs + self.n_cons, self.n_params), 
            device=self.device, dtype=torch.float32)

        
    def step(self, get_obj_con_kl, states_tensor, betas_tensor, mu_kl=0.0):
        # for adaptive kl
        max_kl = self._getMaxKL(mu_kl)

        # calculate gradient
        objectives, constraints, kl = get_obj_con_kl()
        self._computeKLGrad(kl)

        for obj_idx in range(self.n_objs):
            b_tensor = flatGrad(-objectives[obj_idx], self.actor.parameters(), retain_graph=True)
            H_inv_b_tensor = self._conjugateGradient(b_tensor)
            approx_b_tensor = self._Hx(H_inv_b_tensor)
            self.B_tensor[obj_idx].data.copy_(approx_b_tensor)
            self.H_inv_B_tensor[obj_idx].data.copy_(H_inv_b_tensor)

        con_vals = []
        safety_mode = False
        for con_idx in range(self.n_cons):
            b_tensor = flatGrad(constraints[con_idx], self.actor.parameters(), retain_graph=True)
            H_inv_b_tensor = self._conjugateGradient(b_tensor)
            approx_b_tensor = self._Hx(H_inv_b_tensor)
            self.B_tensor[self.n_objs + con_idx].data.copy_(approx_b_tensor)
            self.H_inv_B_tensor[self.n_objs + con_idx].data.copy_(H_inv_b_tensor)
            con_vals.append(constraints[con_idx].item())
            if con_idx == self.n_cons - 1 and self.con_entropy:
                continue
            if con_vals[con_idx] > self.con_thresholds[con_idx]:
                safety_mode = True

        with torch.no_grad():
            # constrct QP problem
            S_tensor = self.B_tensor@self.H_inv_B_tensor.T
            S_mat = S_tensor.detach().cpu().numpy().astype(np.float64)
            S_mat = 0.5*(S_mat + S_mat.T)
            # ===== to ensure S_mat is invertible ===== #
            min_eig_val = min(0.0, np.min(np.linalg.eigvals(S_mat)))
            S_mat += np.eye(S_mat.shape[0])*(-min_eig_val + EPS)
            # ========================================= #
            c_scalars = []
            safe_c_scalars = []
            active_indices = []
            safe_active_indices = []
            for obj_idx in range(self.n_objs):
                b_H_inv_b = S_mat[obj_idx, obj_idx]
                c_scalar = np.sqrt(2.0*max_kl*np.clip(b_H_inv_b, 0.0, np.inf))
                c_scalars.append(c_scalar)
                active_indices.append(obj_idx)
            for con_idx in range(self.n_cons):
                b_H_inv_b = S_mat[self.n_objs + con_idx, self.n_objs + con_idx]
                c_scalar = np.sqrt(2.0*max_kl*np.clip(b_H_inv_b, 0.0, np.inf))
                const_value = con_vals[con_idx] - self.con_thresholds[con_idx]
                if self.con_entropy and con_idx == self.n_cons - 1:
                    if const_value + c_scalar >= 0.0: # for active constraint
                        active_indices.append(self.n_objs + con_idx)
                        safe_active_indices.append(con_idx)
                        safe_c_scalar = min(const_value, c_scalar)
                else:
                    if const_value > 0.0: # for violated constraint
                        safe_active_indices.append(con_idx)
                        safe_c_scalar = c_scalar
                    elif const_value + c_scalar >= 0.0: # for active constraint
                        active_indices.append(self.n_objs + con_idx)
                        safe_active_indices.append(con_idx)
                        safe_c_scalar = min(const_value, c_scalar)
                    else: # for inactive constraint
                        safe_c_scalar = const_value
                safe_c_scalars.append(safe_c_scalar)
                c_scalars.append(const_value)
            c_vector = np.array(c_scalars)
            safe_c_vector = np.array(safe_c_scalars)

            # solve QP
            if not safety_mode:
                try:
                    # solve QP with constraints and objectives
                    temp_S_mat = S_mat[active_indices][:, active_indices]
                    temp_c_vector = c_vector[active_indices]
                    temp_con_lam_vector = solve_qp(
                        P=temp_S_mat, q=-temp_c_vector, lb=np.zeros_like(temp_c_vector))
                    assert temp_con_lam_vector is not None
                    con_lam_vector = np.zeros_like(c_vector)
                    for idx, active_idx in enumerate(active_indices):
                        con_lam_vector[active_idx] = temp_con_lam_vector[idx]

                    # find scaling factor
                    approx_kl = 0.5*np.dot(con_lam_vector, S_mat@con_lam_vector)
                    scaling_factor = min(np.sqrt(max_kl/approx_kl), 1.0)

                    # find search direction
                    lam_tensor = torch.tensor(con_lam_vector, device=self.device, dtype=torch.float32)
                    delta_theta = scaling_factor*(self.H_inv_B_tensor.T@lam_tensor)
                except:
                    # if QP solver failed, then use safety mode
                    print("QP solver failed.")
                    safety_mode = True

            if safety_mode:
                # solve QP with only constraints
                S_mat = S_mat[self.n_objs:, self.n_objs:]
                temp_S_mat = S_mat[safe_active_indices][:, safe_active_indices]
                temp_c_vector = safe_c_vector[safe_active_indices]
                temp_con_lam_vector = solve_qp(
                    P=temp_S_mat, q=-temp_c_vector, lb=np.zeros_like(temp_c_vector))
                assert temp_con_lam_vector is not None, "QP solver failed 2."
                con_lam_vector = np.zeros_like(safe_c_vector)
                for idx, active_idx in enumerate(safe_active_indices):
                    con_lam_vector[active_idx] = temp_con_lam_vector[idx]

                # find scaling factor
                approx_kl = 0.5*np.dot(con_lam_vector, S_mat@con_lam_vector)
                scaling_factor = min(np.sqrt(max_kl/approx_kl), 1.0)

                # find search direction
                lam_tensor = torch.tensor(con_lam_vector, device=self.device, dtype=torch.float32)
                delta_theta = scaling_factor*(self.H_inv_B_tensor[self.n_objs:].T@lam_tensor)

            # backup parameters
            init_theta = torch.cat([t.view(-1) for t in self.actor.parameters()]).clone().detach()

            # update distribution list
            self._applyParams(init_theta - delta_theta)
            means, log_stds, stds = self.actor(states_tensor, betas_tensor)

            # restore parameters
            self._applyParams(init_theta)

        return objectives, constraints, delta_theta
