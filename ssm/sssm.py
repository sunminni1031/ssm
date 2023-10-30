import copy
import warnings
from tqdm.auto import trange
import numba
import numpy as np
from scipy.special import logsumexp
from ssm.util import ssm_pbar
from ssm.messages import gaussian_logpdf


class SSSM(object):
    def __init__(self):
        # number of states
        self.K = 2
        # observation and latent variable dimension
        self.D = 2
        # state-space paramters
        self.m0 = None
        self.sgm0 = None
        self.a_m = None
        self.sigma_m = None
        self.a_v = None
        self.sigma_v = None
        # input-determined switching process paramters
        self.log_p0 = None
        self.log_tmtrx_io = None

    def initialize_params(self):
        # state-space parameters
        self.m0 = np.zeros((self.K, 2))
        self.sgm0 = np.ones(self.K) * 0.5
        self.a_m = np.ones(self.K) * 0.5
        self.sigma_m = np.ones(self.K) * 0.5
        self.a_v = np.ones(self.K) * 0.5
        self.sigma_v = np.ones(self.K) * 0.5
        # input-determined switching process paramters
        self.log_p0 = -np.log(self.K) * np.ones(self.K)
        prob_011 = self.rand_state.uniform(low=0.1, high=0.9)
        prob_100 = self.rand_state.uniform(low=0.1, high=0.3)
        self.log_tmtrx_io = np.array([[[0, -np.inf], [np.log(1 - prob_011), np.log(prob_011)]],
                                      [[np.log(prob_100), np.log(1 - prob_100)], [-np.inf, 0]]])
        return

    @numba.jit(nopython=True, cache=True)
    def kalman_filter(self, obsv, inpt, k, h_t):
        m0 = self.m0[k]
        sgm0 = self.sgm0[k]
        a_m = self.a_m[k]
        sigma_m = self.sigma_m[k]
        a_v = self.a_v[k]
        sigma_v = self.sigma_v[k]
        T, D = obsv.shape
        predicted_mus = np.zeros((T, D))  # preds E[x_t | y_{1:t-1}]
        predicted_sigmas = np.zeros(T)  # preds Cov[x_t | y_{1:t-1}]
        filtered_mus = np.zeros((T, D))  # means E[x_t | y_{1:t}]
        filtered_sigmas = np.zeros(T)  # means Cov[x_t | y_{1:t}]
        # Initialize
        predicted_mus[0] = m0
        predicted_sigmas[0] = sgm0
        ll = 0
        for t in range(T):
            # update the log likelihood
            ll += h_t[t] * gaussian_logpdf(obsv[t] - a_v * obsv[t - 1],
                                           (1 - a_v) * predicted_mus[t],
                                           ((1 - a_v) * predicted_sigmas[t] * (1 - a_v) + sigma_v) * np.eye(D))
            # condition on this frame'sgm observations
            m, sgm = predicted_mus[t], predicted_sigmas[t]
            kalman = (h_t[t] * sgm * (1 - a_v)) / (h_t[t] * (1 - a_v) * sgm * (1 - a_v) + sigma_v)
            obsv_pre = obsv[t - 1] if t > 0 else obsv[0]
            filtered_mus[t] = m + kalman * (obsv[t] - a_v * obsv_pre - (1 - a_v) * m)
            filtered_sigmas[t] = (1 - kalman * (1 - a_v)) * sgm
            if t == T - 1:
                break
            # predict
            xflag = {0: -1, 1: 0, -1: 1}[inpt[t + 1] - inpt[t]]
            a_m_t = a_m if k == xflag else 1.
            sigma_m_t = sigma_m if k == xflag else 0.
            m, sgm = filtered_mus[t], filtered_sigmas[t]
            predicted_mus[t + 1] = a_m_t * m + (1 - a_m_t) * obsv[t]
            predicted_sigmas[t + 1] = a_m_t * sgm * a_m_t + sigma_m_t
        return filtered_mus, filtered_sigmas, predicted_mus, predicted_sigmas, ll

    @numba.jit(nopython=True, cache=True)
    def kalman_smoother(self, obsv, inpt, k, h_t):
        T, D = obsv.shape
        filtered_mus, filtered_sigmas, predicted_mus, predicted_sigmas, ll = self.kalman_filter(obsv, inpt, k, h_t)
        smoothed_mus = np.zeros((T, D))
        smoothed_sigmas = np.zeros(T)
        E_cross = np.zeros((T - 1, D, D))

        smoothed_mus[-1] = filtered_mus[-1]
        smoothed_sigmas[-1] = filtered_sigmas[-1]

        for t in range(T - 2, -1, -1):
            xflag = {0: -1, 1: 0, -1: 1}[inpt[t + 1] - inpt[t]]
            a_m_t = self.a_m[k] if k == xflag else 1.
            gt = filtered_sigmas[t] * a_m_t / predicted_sigmas[t + 1]
            smoothed_mus[t] = filtered_mus[t] + gt * (smoothed_mus[t + 1] - predicted_mus[t + 1])
            smoothed_sigmas[t] = filtered_sigmas[t] + gt * (smoothed_sigmas[t + 1] - predicted_sigmas[t + 1]) * gt
            E_cross[t] = gt * smoothed_sigmas[t + 1] * np.eye(D) + np.outer(smoothed_mus[t], smoothed_mus[t + 1])
        return smoothed_mus, smoothed_sigmas, E_cross, ll

    def compute_q_tk(self, obsv, inpt, h_tk):
        q_tk = np.zeros(h_tk.shape)
        Estats_c = [None for _ in self.K]
        norms_c = np.zeros(self.K)
        for k in range(self.K):
            smoothed_mus, smoothed_sigmas, E_cross, norm_c = self.kalman_smoother(obsv, inpt, k, h_tk[k])
            obsv_pre = np.concatenate([obsv[0][None, :], obsv[:-1]], axis=0)
            v_err = obsv - self.a_v[k] * obsv_pre
            log_qt = (-1/2) * np.einsum('td,td->t', v_err, v_err)
            log_qt += np.einsum('td,td->t', v_err, (1 - self.a_v[k]) * smoothed_mus)
            log_qt += (-1/2) * np.square(1 - self.a_v[k]) * smoothed_sigmas * 2
            log_qt = log_qt / self.sigma_v[k] - np.log(2*np.pi) - np.log(self.sigma_v[k])
            q_tk[:, k] = np.exp(log_qt)
            Estats_c[k] = (smoothed_mus, smoothed_sigmas, E_cross)
            norms_c[k] = norm_c
        return q_tk, Estats_c, norms_c

    def forward_backward(self, obsv, inpt, q_tk):
        T = len(obsv)
        log_likes = np.log(q_tk)
        log_tmtrx_t = np.empty((T - 1, self.K, self.K))
        for t in range(1, T):
            log_tmtrx_t[t - 1] = self.log_tmtrx_io[int(inpt[t, 0])]
        tmtrx_t = np.exp(log_tmtrx_t)
        # forward
        alphas = np.zeros((T, self.K))
        alphas[0] = self.log_p0 + log_likes[0]
        for t in range(T - 1):
            m = np.max(alphas[t])
            alphas[t + 1] = np.log(np.dot(np.exp(alphas[t] - m), tmtrx_t[t])) + m + log_likes[t + 1]
        normalizer = logsumexp(alphas[-1])
        # backward
        betas = np.zeros((T, self.K))
        betas[T - 1] = 0
        for t in range(T - 2, -1, -1):
            tmp = log_likes[t + 1] + betas[t + 1]
            m = np.max(tmp)
            betas[t] = np.log(np.dot(tmtrx_t[t], np.exp(tmp - m))) + m
        # expected states
        expected_states = alphas + betas
        expected_states -= logsumexp(expected_states, axis=1, keepdims=True)
        expected_states = np.exp(expected_states)
        # expected joints
        expected_joints = alphas[:-1, :, None] + betas[1:, None, :] + log_likes[1:, None, :] + log_tmtrx_t
        expected_joints -= expected_joints.max((1, 2))[:, None, None]
        expected_joints = np.exp(expected_joints)
        expected_joints /= expected_joints.sum((1, 2))[:, None, None]
        return expected_states, expected_joints, normalizer

    def compute_h_tk(self, obsv, inpt, q_tk):
        h_tk, Ejoints_d, norm_d = self.forward_backward(obsv, inpt, q_tk)
        Estats_d = (h_tk, Ejoints_d)
        return Estats_d, norm_d

    def mstep_state_space(self, obsvs, inpts, h_tk_list, Estats_c_list):
        for k in range(self.K):
            m0_k, sgm0_k, cnt_0 = np.zeros(self.D), 0, 0
            x_term_m, sq_term0_m, sq_term1_m, cnt_m = 0, 0, 0, 0
            x_term_v, sq_term0_v, sq_term1_v, cnt_v = 0, 0, 0, 0
            for idata in range(len(obsvs)):
                obsv, inpt = obsvs[idata], inpts[idata]
                h_t = h_tk_list[idata][:, k]
                mus, sigmas, E_cross = Estats_c_list[idata][k]
                E_outer = np.einsum('t,ij->tij', sigmas, np.eye(self.D))
                E_outer += np.einsum('ti,tj->tij', mus, mus)
                m0_k += mus[0]
                sgm0_k += 2 * sigmas[0] + mus[0].T @ mus[0]
                cnt_0 += 1
                for t in range(len(obsv) - 1):
                    xflag = {0: -1, 1: 0, -1: 1}[inpt[t + 1] - inpt[t]]
                    if k == xflag:
                        x_term_m += np.trace(E_cross[t]) - (mus[t] + mus[t + 1]).T @ obsv[t] + obsv[t].T @ obsv[t]
                        sq_term0_m += np.trace(E_outer[t]) - 2 * mus[t].T @ obsv[t] + obsv[t].T @ obsv[t]
                        sq_term1_m += np.trace(E_outer[t+1]) - 2 * mus[t + 1].T @ obsv[t] + obsv[t].T @ obsv[t]
                        cnt_m += 1
                for t in range(len(obsv) - 1):
                    x_term_v += h_t[t] * (
                                obsv[t + 1].T @ obsv[t] - (obsv[t + 1] + obsv[t]).T @ mus[t + 1] + np.trace(E_outer[t+1]))
                    sq_term0_v += h_t[t] * (obsv[t].T @ obsv[t] - 2 * obsv[t].T @ mus[t + 1] + np.trace(E_outer[t+1]))
                    sq_term1_v += h_t[t] * (
                                obsv[t + 1].T @ obsv[t + 1] - 2 * obsv[t + 1].T @ mus[t + 1] + np.trace(E_outer[t+1]))
                    cnt_v += h_t[t]
            self.m0[k] = m0_k / cnt_0
            self.sgm0[k] = sgm0_k / cnt_0 / 2 - self.m0[k].T @ self.m0[k] / 2
            self.a_m[k] = x_term_m / sq_term0_m
            self.sigma_m[k] = (sq_term1_m - 2 * self.a_m[k] * x_term_m + sq_term0_m * np.square(
                self.a_m[k])) / cnt_m / 2
            self.a_v[k] = x_term_v / sq_term0_v
            self.sigma_v[k] = (sq_term1_v - 2 * self.a_v[k] * x_term_v + sq_term0_v * np.square(
                self.a_v[k])) / cnt_v / 2
        return

    def mstep_switch(self, obsvs, inpts, q_tk_list, Estats_d_list):
        p0 = 1e-8
        tmtrx_io = [1e-32, 1e-32]
        for obsv, inpt, q_tk, Estats_d in zip(obsvs, inpts, q_tk_list, Estats_d_list):
            expected_states, expected_joints = Estats_d
            p0 += expected_states[0]
            for io in range(self.K):
                tmtrx_io[io] += np.sum(expected_joints[inpt[1:, 0] == io], axis=0)
        self.log_p0 = np.log(p0 / p0.sum())
        for io in range(self.K):
            tmtrx = tmtrx_io[io]
            tmtrx = np.nan_to_num(tmtrx / tmtrx.sum(axis=-1, keepdims=True))
            tmtrx = np.where(tmtrx.sum(axis=-1, keepdims=True) == 0, 1.0 / self.K, tmtrx)
            log_tmtrx = np.log(tmtrx)
            self.log_tmtrx_io[io] = log_tmtrx - logsumexp(log_tmtrx, axis=-1, keepdims=True)
        return

    def fit(self, obsvs, inpts,
            num_iters_em=100, tol_em=0.01,
            verbose=2, learning=True):
        if learning:
            self.initialize_params()
        # initialize variational parameters
        ndata = len(obsvs)
        h_tk_list = [None for _ in range(ndata)]
        q_tk_list = [None for _ in range(ndata)]
        Estats_c_list = [None for _ in range(ndata)]
        Estats_d_list = [None for _ in range(ndata)]

        for idata in range(ndata):
            obsv, inpt = obsvs[idata], inpt[idata]
            h_tk = np.zeros((len(inpt), max(inpt) + 1))
            h_tk[np.arange(len(inpt)), inpt] = 1
            Estats_d_list[idata] = (h_tk, None)

        elbos = []
        # iteration
        pbar = ssm_pbar(num_iters_em, verbose, "ELBO: {:.1f}", [0])
        for itr_em in pbar:
            elbo = 0
            for idata in range(ndata):
                obsv, inpt = obsvs[idata], inpt[idata]
                h_tk = Estats_d_list[idata][0]
                q_tk, Estats_c, norms_c = self.compute_q_tk(obsv, inpt, h_tk)
                Estats_d, norm_d = self.compute_h_tk(obsv, inpt, q_tk)
                elbo += - np.sum(h_tk * np.log(q_tk)) + np.sum(norms_c) + norm_d
                h_tk_list[idata] = h_tk
                Estats_c_list[idata] = Estats_c
                q_tk_list[idata] = q_tk
                Estats_d_list[idata] = Estats_d
            elbos.append(elbo)

            if learning:
                self.mstep_state_space(obsvs, inpts, h_tk_list, Estats_c_list)
                self.mstep_switch(obsvs, inpts, q_tk_list, Estats_d_list)

            if verbose == 2:
                pbar.set_description("ELBO: {:.1f}".format(elbos[-1]))
            # if itr_em > 0 and np.abs(elbos[-1] - elbos[-2]) < tol_em:
            #     if verbose == 2:
            #         pbar.set_description("Converged to LP: {:.1f}".format(elbos[-1]))
            #     break
        return np.array(elbos)
