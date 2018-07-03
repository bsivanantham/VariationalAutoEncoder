import numpy as np
from .mathutils import sigmoid, sigmoid_prime, tanh_prime


class NoOutputLstm:
    def __init__(self, input_size: int, hidden_size: int):
        self.w_xf_g = np.random.randn(input_size, hidden_size)
        self.w_hf_g = np.random.randn(hidden_size, hidden_size)
        self.b_f_g = np.random.standard_normal(hidden_size)
        self.w_xi_g = np.random.randn(input_size, hidden_size)
        self.w_hi_g = np.random.randn(hidden_size, hidden_size)
        self.b_i_g = np.random.standard_normal(hidden_size)
        self.w_xc = np.random.randn(input_size, hidden_size)
        self.w_hc = np.random.randn(hidden_size, hidden_size)
        self.b_c = np.random.standard_normal(hidden_size)

    def clone(self):
        clone = NoOutputLstm(0, 0)
        clone.w_xf_g = np.copy(self.w_xf_g)
        clone.w_hf_g = np.copy(self.w_hf_g)
        clone.b_f_g = np.copy(self.b_f_g)
        clone.w_xi_g = np.copy(self.w_xi_g)
        clone.w_hi_g = np.copy(self.w_hi_g)
        clone.b_i_g = np.copy(self.b_i_g)
        clone.w_xc = np.copy(self.w_xc)
        clone.w_hc = np.copy(self.w_hc)
        clone.b_c = np.copy(self.b_c)

        return clone

    def _step(self, x, h_prev):
        f_g = sigmoid(np.dot(x, self.w_xf_g) + np.dot(h_prev, self.w_hf_g) + self.b_f_g)
        i_g = sigmoid(np.dot(x, self.w_xi_g) + np.dot(h_prev, self.w_hi_g) + self.b_i_g)
        c = np.tanh(np.dot(x, self.w_xc) + np.dot(h_prev, self.w_hc) + self.b_c)
        h = h_prev * f_g + i_g * c

        return h, f_g, i_g, c

    def _forward_prop(self, xs, h0):
        hs, f_gs, i_gs, cs = [], [], [], []
        h = h0
        for i, x in enumerate(xs):
            hs.append(h)
            h, f_g, i_g, c = self._step(x, h)
            f_gs.append(f_g)
            i_gs.append(i_g)
            cs.append(c)
        return np.asarray(hs), np.asarray(f_gs), np.asarray(i_gs), np.asarray(cs), h

    def _back_step(self, x, h_prev, f_g, i_g, c, dh):
        df_g = dh * h_prev
        dact_f_g = df_g * sigmoid_prime(f_g)
        dw_xf_g = dact_f_g * np.expand_dims(x, axis=1)
        dw_hf_g = dact_f_g * np.expand_dims(h_prev, axis=1)
        db_f_g = dact_f_g

        dh_prev = np.dot(self.w_hf_g, dact_f_g)

        di = dh * c
        dact_i_g = di * sigmoid_prime(i_g)
        dw_xi_g = dact_i_g * np.expand_dims(x, axis=1)
        dw_hi_g = dact_i_g * np.expand_dims(h_prev, axis=1)
        db_i_g = dact_i_g

        dh_prev += np.dot(self.w_hi_g, dact_i_g)

        dc = dh * i_g
        dact_c = dc * tanh_prime(c)
        dw_xc = dact_c * np.expand_dims(x, axis=1)
        dw_hc = dact_c * np.expand_dims(h_prev, axis=1)
        db_c = dact_c

        dh_prev += np.dot(self.w_hc, dact_c)

        dh_prev += dh * f_g

        return dw_xf_g, dw_hf_g, db_f_g, dw_xi_g, dw_hi_g, db_i_g, dw_xc, dw_hc, db_c, dh_prev

    def _back_prop(self, xs, hs, f_gs, i_gs, cs, dh_last):
        delta_w_xf_g = np.zeros(self.w_xf_g.shape)
        delta_w_hf_g = np.zeros(self.w_hf_g.shape)
        delta_b_f_g = np.zeros(self.b_f_g.shape)
        delta_w_xi_g = np.zeros(self.w_xi_g.shape)
        delta_w_hi_g = np.zeros(self.w_hi_g.shape)
        delta_b_i_g = np.zeros(self.b_i_g.shape)
        delta_w_xc = np.zeros(self.w_xc.shape)
        delta_w_hc = np.zeros(self.w_hc.shape)
        delta_b_c = np.zeros(self.b_c.shape)

        dh = dh_last
        for x, h_prev, f_g, i_g, c in reversed(list(zip(xs, hs, f_gs, i_gs, cs))):
            dw_xf_g, dw_hf_g, db_f_g, dw_xi_g, dw_hi_g, db_i_g, dw_xc, dw_hc, db_c, dh_prev = self._back_step(x, h_prev, f_g, i_g, c, dh)

            delta_w_xf_g += dw_xf_g
            delta_w_hf_g += dw_hf_g
            delta_b_f_g += db_f_g
            delta_w_xi_g += dw_xi_g
            delta_w_hi_g += dw_hi_g
            delta_b_i_g += db_i_g
            delta_w_xc += dw_xc
            delta_w_hc += dw_hc
            delta_b_c += db_c

            dh = dh_prev

        return delta_w_xf_g, delta_w_hf_g, delta_b_f_g, delta_w_xi_g, delta_w_hi_g, delta_b_i_g, delta_w_xc, delta_w_hc, delta_b_c

    def activate(self, xs, h0):
        _, _, _, _, h = self._forward_prop(xs, h0)
        return h

    def train(self, xs, h0, t, learning_rate):
        hs, f_gs, i_gs, cs, h = self._forward_prop(xs, h0)
        dh = h - t
        dw_xf_g, dw_hf_g, db_f_g, dw_xi_g, dw_hi_g, db_i_g, dw_xc, dw_hc, db_c = self._back_prop(xs, hs, f_gs, i_gs, cs, dh)
        self.w_xf_g -= dw_xf_g * learning_rate
        self.w_hf_g -= dw_hf_g * learning_rate
        self.b_f_g -= db_f_g * learning_rate
        self.w_xi_g -= dw_xi_g * learning_rate
        self.w_hi_g -= dw_hi_g * learning_rate
        self.b_i_g -= db_i_g * learning_rate
        self.w_xc -= dw_xc * learning_rate
        self.w_hc -= dw_hc * learning_rate
        self.b_c -= db_c * learning_rate
