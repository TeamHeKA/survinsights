import os, random, numpy as np, tensorflow as tf
SEED = 123
os.environ["PYTHONHASHSEED"] = str(SEED)
random.seed(SEED)
np.random.seed(SEED)
tf.random.set_seed(SEED)
tf.keras.utils.set_random_seed(SEED)
tf.config.experimental.enable_op_determinism()

_EPSILON = 1e-08

##### USER-DEFINED FUNCTIONS (TF2)
def log(x):
    return tf.math.log(x + _EPSILON)

def div(x, y):
    return x / (y + _EPSILON)

def get_seq_length(sequence):
    # sequence: [B, T, D]
    used = tf.sign(tf.reduce_max(tf.abs(sequence), axis=2))  # [B, T]
    tmp_length = tf.reduce_sum(used, axis=1)                 # [B]
    return tf.cast(tmp_length, tf.int32)


class Model_Longitudinal_Attention(tf.keras.Model):
    """
    TF2 version of the TF1 Model_Longitudinal_Attention.

    Keeps the same public method names so existing training scripts can be updated minimally.
    """

    def __init__(self, sess, name, input_dims, network_settings):
        super().__init__(name=name)
        self.sess = sess  # kept for backward-compat; unused in TF2
        self.name_ = name

        # INPUT DIMENSIONS
        self.x_dim        = input_dims['x_dim']
        self.x_dim_cont   = input_dims['x_dim_cont']
        self.x_dim_bin    = input_dims['x_dim_bin']
        self.num_Event    = input_dims['num_Event']
        self.num_Category = input_dims['num_Category']
        self.max_length   = input_dims['max_length']

        # NETWORK HYPER-PARAMETERS
        self.h_dim1         = network_settings['h_dim_RNN']
        self.h_dim2         = network_settings['h_dim_FC']
        self.num_layers_RNN = network_settings['num_layers_RNN']
        self.num_layers_ATT = network_settings['num_layers_ATT']
        self.num_layers_CS  = network_settings['num_layers_CS']

        self.RNN_type      = network_settings['RNN_type']          # e.g. "LSTM" or "GRU"
        self.FC_active_fn  = network_settings['FC_active_fn']       # callable, e.g. tf.nn.relu
        self.RNN_active_fn = network_settings['RNN_active_fn']      # kept for API similarity
        self.initial_W     = network_settings['initial_W']          # can be keras initializer or None

        self.reg_W     = tf.keras.regularizers.L1(network_settings['reg_W'])
        self.reg_W_out = tf.keras.regularizers.L1(network_settings['reg_W_out'])

        # Optimizers (created here; LR is passed per-step)
        self.optimizer = tf.keras.optimizers.Adam()
        self.optimizer_burn_in = tf.keras.optimizers.Adam()

        self._build_layers()

        # Exposed "tensor-like" attributes for predict_* functions (populated per forward call)
        self.out = None
        self.z = None
        self.z_mean = None
        self.z_std = None
        self.rnn_final_state = None
        self.att_weight = None
        self.context_vec = None

        # Loss scalars (populated per step)
        self.LOSS_1 = None
        self.LOSS_2 = None
        self.LOSS_3 = None
        self.LOSS_TOTAL = None
        self.LOSS_BURNIN = None
    
    def call(self, inputs, training=False, keep_prob=1.0):
        x, x_mi = inputs
        # run your existing forward to create variables and set attributes
        _ = self.forward(x, x_mi, training=training, keep_prob=keep_prob)
        return self.out

    def _build_layers(self):
        # --- RNN stack ---
        self.rnn_layers = []
        for i in range(self.num_layers_RNN):
            if str(self.RNN_type).upper() == "LSTM":
                self.rnn_layers.append(
                    tf.keras.layers.LSTM(
                        self.h_dim1,
                        return_sequences=True,
                        return_state=True,
                        name=f"rnn_lstm_{i}"
                    )
                )
            else:
                # default to GRU
                self.rnn_layers.append(
                    tf.keras.layers.GRU(
                        self.h_dim1,
                        return_sequences=True,
                        return_state=True,
                        name=f"rnn_gru_{i}"
                    )
                )

        # --- Attention MLP: score([h_j, all_last]) -> scalar ---
        att_layers = []
        for i in range(self.num_layers_ATT):
            att_layers.append(tf.keras.layers.Dense(
                self.h_dim2,
                activation=tf.nn.tanh,
                kernel_initializer=self.initial_W,
                name=f"att_fc_{i}"
            ))
        att_layers.append(tf.keras.layers.Dense(
            1,
            activation=None,
            kernel_initializer=self.initial_W,
            name="att_out"
        ))
        self.att_mlp = tf.keras.Sequential(att_layers, name="att_mlp")

        # --- RNN output heads for z_mean, z_std ---
        self.z_mean_head = tf.keras.layers.Dense(
            self.x_dim,
            activation=None,
            kernel_initializer=self.initial_W,
            name="RNN_out_mean1"
        )
        self.z_std_head = tf.keras.layers.Dense(
            self.x_dim,
            activation=None,
            kernel_initializer=self.initial_W,
            name="RNN_out_std1"
        )

        # --- Cause-specific network ---
        self.combine_fc = tf.keras.layers.Dense(
            self.h_dim2,
            activation=self.FC_active_fn,
            kernel_initializer=self.initial_W,
            kernel_regularizer=self.reg_W,
            name="Layer1"
        )

        # One subnetwork per event (each outputs h_dim2 features)
        self.cs_nets = []
        for e in range(self.num_Event):
            layers = []
            for i in range(self.num_layers_CS):
                layers.append(tf.keras.layers.Dense(
                    self.h_dim2,
                    activation=self.FC_active_fn,
                    kernel_initializer=self.initial_W,
                    kernel_regularizer=self.reg_W,
                    name=f"cs{e}_fc_{i}"
                ))
            self.cs_nets.append(tf.keras.Sequential(layers, name=f"cs_net_{e}"))

        self.final_out = tf.keras.layers.Dense(
            self.num_Event * self.num_Category,
            activation=tf.nn.softmax,
            kernel_initializer=self.initial_W,
            kernel_regularizer=self.reg_W_out,
            name="Output"
        )

    def _extract_last_and_hist(self, x, x_mi):
        """
        Replicates TF1 logic:
        - determine last observed time (via non-zero padding)
        - build masks
        - x_last/mi_last = last time step's measurement without delta col (slice [1:])
        - x_hist/mi_hist = history excluding last step (length max_length-1)
        """
        # seq_length based on x (original did on self.x)
        seq_length = get_seq_length(x)  # [B]

        tmp_range = tf.expand_dims(tf.range(0, self.max_length, 1), axis=0)  # [1,T]
        rnn_mask1 = tf.cast(tmp_range <= tf.expand_dims(seq_length - 1, axis=1), tf.float32)  # [B,T]
        rnn_mask2 = tf.cast(tmp_range == tf.expand_dims(seq_length - 1, axis=1), tf.float32)  # [B,T]

        # last measurement (sum over time using rnn_mask2)
        x_last_full = tf.reduce_sum(tf.expand_dims(rnn_mask2, axis=2) * x, axis=1)     # [B, x_dim]
        x_last = x_last_full[:, 1:]  # remove delta col

        x_hist_full = x * (1.0 - tf.expand_dims(rnn_mask2, axis=2))
        x_hist = x_hist_full[:, :(self.max_length - 1), :]  # [B, T-1, x_dim]

        mi_last_full = tf.reduce_sum(tf.expand_dims(rnn_mask2, axis=2) * x_mi, axis=1)  # [B, x_dim]
        mi_last = mi_last_full[:, 1:]

        mi_hist_full = x_mi * (1.0 - tf.expand_dims(rnn_mask2, axis=2))
        mi_hist = mi_hist_full[:, :(self.max_length - 1), :]

        return x_last, x_hist, mi_last, mi_hist, rnn_mask1, rnn_mask2

    def _run_rnn(self, all_hist, training, keep_prob):
        """
        Run stacked RNN layers. Returns:
        - rnn_outputs: [B, T-1, h_dim1] (last layer outputs)
        - final_state: tuple/list depending on RNN type and layers (stored for API)
        """
        x_in = all_hist
        final_states = []

        for layer in self.rnn_layers:
            if isinstance(layer, tf.keras.layers.LSTM):
                seq_out, h, c = layer(x_in, training=training)
                x_in = seq_out
                final_states.append((h, c))
            else:
                seq_out, h = layer(x_in, training=training)
                x_in = seq_out
                final_states.append(h)

        return x_in, final_states

    def _temporal_attention(self, rnn_states, all_last, rnn_mask_att):
        """
        rnn_states: [B, T-1, H]
        all_last:   [B, 2*x_dim_without_delta]
        rnn_mask_att: [B, T-1] 1 where measured, 0 where not measured
        """
        B = tf.shape(rnn_states)[0]
        Tm1 = tf.shape(rnn_states)[1]

        all_last_rep = tf.tile(tf.expand_dims(all_last, axis=1), [1, Tm1, 1])  # [B, T-1, *]
        att_in = tf.concat([rnn_states, all_last_rep], axis=2)                # [B, T-1, H+*]

        e = self.att_mlp(att_in)                     # [B, T-1, 1]
        e = tf.exp(e)
        e = tf.squeeze(e, axis=2)                    # [B, T-1]
        e = e * rnn_mask_att                         # mask unmeasured

        a = div(e, tf.reduce_sum(e, axis=1, keepdims=True) + _EPSILON)  # [B, T-1]

        context = tf.reduce_sum(
            tf.expand_dims(a, axis=2) * rnn_states,
            axis=1
        )  # [B, H]

        return a, context

    def forward(self, x, x_mi, training, keep_prob):
        """
        One forward pass. Populates self.out, self.z, etc.
        """
        x_last, x_hist, mi_last, mi_hist, rnn_mask1, rnn_mask2 = self._extract_last_and_hist(x, x_mi)

        all_hist = tf.concat([x_hist, mi_hist], axis=2)  # [B, T-1, 2*x_dim]
        all_last = tf.concat([x_last, mi_last], axis=1)  # [B, 2*(x_dim-1)]

        # Attention mask (measured timestamps in x_hist)
        rnn_mask_att = tf.cast(tf.not_equal(tf.reduce_sum(x_hist, axis=2), 0.0), tf.float32)  # [B, T-1]

        # RNN outputs and hidden states
        rnn_outputs, final_states = self._run_rnn(all_hist, training=training, keep_prob=keep_prob)
        rnn_states = rnn_outputs  # use last-layer sequence output as h_j

        # Temporal attention
        att_weight, context_vec = self._temporal_attention(rnn_states, all_last, rnn_mask_att)

        # z mean/std for prediction loss
        z_mean = self.z_mean_head(rnn_outputs)                      # [B, T-1, x_dim]
        z_std = tf.exp(self.z_std_head(rnn_outputs))                # [B, T-1, x_dim]
        eps = tf.random.normal(tf.shape(z_mean), mean=0.0, stddev=1.0, dtype=tf.float32)
        z = z_mean + z_std * eps

        # Cause-specific branch
        inputs = tf.concat([x_last, context_vec], axis=1)  # [B, (x_dim-1) + H]
        h = self.combine_fc(inputs)
        h = tf.nn.dropout(h, rate=(1.0 - keep_prob)) if training else h

        out_list = []
        for e in range(self.num_Event):
            cs_h = self.cs_nets[e](h, training=training)
            out_list.append(cs_h)

        out = tf.stack(out_list, axis=1)                      # [B, E, h_dim2]
        out = tf.reshape(out, [-1, self.num_Event * self.h_dim2])
        out = tf.nn.dropout(out, rate=(1.0 - keep_prob)) if training else out

        out = self.final_out(out)                             # [B, E*C]
        out = tf.reshape(out, [-1, self.num_Event, self.num_Category])

        # Store for predict_* API
        self.out = out
        self.z = z
        self.z_mean = z_mean
        self.z_std = z_std
        self.rnn_final_state = final_states
        self.att_weight = att_weight
        self.context_vec = context_vec

        # Also return masks needed for LOSS_3
        return {
            "out": out,
            "z": z,
            "z_mean": z_mean,
            "z_std": z_std,
            "rnn_mask1": rnn_mask1,
        }

    ### LOSS-FUNCTION 1 -- Log-likelihood loss
    def loss_Log_Likelihood(self, k, fc_mask1, fc_mask2, out):
        sigma3 = tf.constant(1.0, dtype=tf.float32)

        I_1 = tf.sign(k)  # [B,1]
        denom = 1.0 - tf.reduce_sum(tf.reduce_sum(fc_mask1 * out, axis=2), axis=1, keepdims=True)
        denom = tf.clip_by_value(denom, _EPSILON, 1.0 - _EPSILON)

        tmp1 = tf.reduce_sum(tf.reduce_sum(fc_mask2 * out, axis=2), axis=1, keepdims=True)
        tmp1 = I_1 * log(div(tmp1, denom))

        tmp2 = tf.reduce_sum(tf.reduce_sum(fc_mask2 * out, axis=2), axis=1, keepdims=True)
        tmp2 = (1.0 - I_1) * log(div(tmp2, denom))

        return -tf.reduce_mean(tmp1 + sigma3 * tmp2)

    ### LOSS-FUNCTION 2 -- Ranking loss
    def loss_Ranking(self, k, t, fc_mask3, out):
        sigma1 = tf.constant(0.1, dtype=tf.float32)

        eta = []
        one_vector = tf.ones_like(t, dtype=tf.float32)  # [B,1]

        for e in range(self.num_Event):
            I_2 = tf.cast(tf.equal(k, e + 1), dtype=tf.float32)  # [B,1]
            I_2 = tf.linalg.diag(tf.squeeze(I_2, axis=1))        # [B,B]

            tmp_e = out[:, e, :]                                 # [B, C]
            R = tf.matmul(tmp_e, tf.transpose(fc_mask3))         # [B, B]

            diag_R = tf.reshape(tf.linalg.diag_part(R), [-1, 1]) # [B,1]
            R = tf.matmul(one_vector, tf.transpose(diag_R)) - R
            R = tf.transpose(R)

            T = tf.nn.relu(
                tf.sign(tf.matmul(one_vector, tf.transpose(t)) - tf.matmul(t, tf.transpose(one_vector)))
            )
            T = tf.matmul(I_2, T)

            tmp_eta = tf.reduce_mean(T * tf.exp(-R / sigma1), axis=1, keepdims=True)
            eta.append(tmp_eta)

        eta = tf.stack(eta, axis=1)  # [B, E, 1]
        eta = tf.reduce_mean(tf.reshape(eta, [-1, self.num_Event]), axis=1, keepdims=True)
        return tf.reduce_sum(eta)

    ### LOSS-FUNCTION 3 -- RNN prediction loss
    def loss_RNN_Prediction(self, x, x_mi, z, rnn_mask1):
        tmp_x  = x[:, 1:, :]      # (t=2 ~ M)
        tmp_mi = x_mi[:, 1:, :]   # (t=2 ~ M)

        tmp_mask1 = tf.tile(tf.expand_dims(rnn_mask1, axis=2), [1, 1, self.x_dim])
        tmp_mask1 = tmp_mask1[:, :(self.max_length - 1), :]

        zeta = tf.reduce_mean(
            tf.reduce_sum(tmp_mask1 * (1.0 - tmp_mi) * tf.pow(z - tmp_x, 2), axis=1)
        )
        return zeta

    def _regularization_loss(self):
        # Keras collects regularization losses from layers automatically in self.losses
        if not self.losses:
            return tf.constant(0.0, dtype=tf.float32)
        return tf.add_n(self.losses)

    def get_cost(self, DATA, MASK, MISSING, PARAMETERS, keep_prob, lr_train):
        (x_mb, k_mb, t_mb)    = DATA
        (m1_mb, m2_mb, m3_mb) = MASK
        (x_mi_mb,)            = MISSING
        (alpha, beta, gamma)  = PARAMETERS

        x_mb = tf.convert_to_tensor(x_mb, dtype=tf.float32)
        x_mi_mb = tf.convert_to_tensor(x_mi_mb, dtype=tf.float32)
        k_mb = tf.convert_to_tensor(k_mb, dtype=tf.float32)
        t_mb = tf.convert_to_tensor(t_mb, dtype=tf.float32)
        m1_mb = tf.convert_to_tensor(m1_mb, dtype=tf.float32)
        m2_mb = tf.convert_to_tensor(m2_mb, dtype=tf.float32)
        m3_mb = tf.convert_to_tensor(m3_mb, dtype=tf.float32)

        fwd = self.forward(x_mb, x_mi_mb, training=False, keep_prob=keep_prob)
        out = fwd["out"]

        L1 = self.loss_Log_Likelihood(k_mb, m1_mb, m2_mb, out)
        L2 = self.loss_Ranking(k_mb, t_mb, m3_mb, out)
        L3 = self.loss_RNN_Prediction(x_mb, x_mi_mb, fwd["z"], fwd["rnn_mask1"])
        reg = self._regularization_loss()

        total = alpha * L1 + beta * L2 + gamma * L3 + reg
        return float(total.numpy())

    def train(self, DATA, MASK, MISSING, PARAMETERS, keep_prob, lr_train):
        (x_mb, k_mb, t_mb)    = DATA
        (m1_mb, m2_mb, m3_mb) = MASK
        (x_mi_mb,)            = MISSING
        (alpha, beta, gamma)  = PARAMETERS

        x_mb = tf.convert_to_tensor(x_mb, dtype=tf.float32)
        x_mi_mb = tf.convert_to_tensor(x_mi_mb, dtype=tf.float32)
        k_mb = tf.convert_to_tensor(k_mb, dtype=tf.float32)
        t_mb = tf.convert_to_tensor(t_mb, dtype=tf.float32)
        m1_mb = tf.convert_to_tensor(m1_mb, dtype=tf.float32)
        m2_mb = tf.convert_to_tensor(m2_mb, dtype=tf.float32)
        m3_mb = tf.convert_to_tensor(m3_mb, dtype=tf.float32)

        # update LR for this step
        self.optimizer.learning_rate = lr_train

        with tf.GradientTape() as tape:
            fwd = self.forward(x_mb, x_mi_mb, training=True, keep_prob=keep_prob)
            out = fwd["out"]

            L1 = self.loss_Log_Likelihood(k_mb, m1_mb, m2_mb, out)
            L2 = self.loss_Ranking(k_mb, t_mb, m3_mb, out)
            L3 = self.loss_RNN_Prediction(x_mb, x_mi_mb, fwd["z"], fwd["rnn_mask1"])
            reg = self._regularization_loss()

            total = alpha * L1 + beta * L2 + gamma * L3 + reg

        grads = tape.gradient(total, self.trainable_variables)
        self.optimizer.apply_gradients(zip(grads, self.trainable_variables))

        # store like TF1 attributes
        self.LOSS_1 = L1
        self.LOSS_2 = L2
        self.LOSS_3 = L3
        self.LOSS_TOTAL = total

        return None, float(total.numpy())

    def train_burn_in(self, DATA, MISSING, keep_prob, lr_train):
        (x_mb, k_mb, t_mb) = DATA
        (x_mi_mb,)         = MISSING

        x_mb = tf.convert_to_tensor(x_mb, dtype=tf.float32)
        x_mi_mb = tf.convert_to_tensor(x_mi_mb, dtype=tf.float32)

        self.optimizer_burn_in.learning_rate = lr_train

        with tf.GradientTape() as tape:
            fwd = self.forward(x_mb, x_mi_mb, training=True, keep_prob=keep_prob)
            L3 = self.loss_RNN_Prediction(x_mb, x_mi_mb, fwd["z"], fwd["rnn_mask1"])
            reg = self._regularization_loss()
            burn = L3 + reg

        grads = tape.gradient(burn, self.trainable_variables)
        self.optimizer_burn_in.apply_gradients(zip(grads, self.trainable_variables))

        self.LOSS_3 = L3
        self.LOSS_BURNIN = burn

        return None, float(L3.numpy())

    def predict(self, x_test, x_mi_test, keep_prob=1.0):
        x_test = tf.convert_to_tensor(x_test, dtype=tf.float32)
        x_mi_test = tf.convert_to_tensor(x_mi_test, dtype=tf.float32)
        self.forward(x_test, x_mi_test, training=False, keep_prob=keep_prob)
        return self.out.numpy()

    def predict_z(self, x_test, x_mi_test, keep_prob=1.0):
        x_test = tf.convert_to_tensor(x_test, dtype=tf.float32)
        x_mi_test = tf.convert_to_tensor(x_mi_test, dtype=tf.float32)
        self.forward(x_test, x_mi_test, training=False, keep_prob=keep_prob)
        return self.z.numpy()

    def predict_rnnstate(self, x_test, x_mi_test, keep_prob=1.0):
        x_test = tf.convert_to_tensor(x_test, dtype=tf.float32)
        x_mi_test = tf.convert_to_tensor(x_mi_test, dtype=tf.float32)
        self.forward(x_test, x_mi_test, training=False, keep_prob=keep_prob)
        # returns Python structure of final states (per layer)
        return self.rnn_final_state

    def predict_att(self, x_test, x_mi_test, keep_prob=1.0):
        x_test = tf.convert_to_tensor(x_test, dtype=tf.float32)
        x_mi_test = tf.convert_to_tensor(x_mi_test, dtype=tf.float32)
        self.forward(x_test, x_mi_test, training=False, keep_prob=keep_prob)
        return self.att_weight.numpy()

    def predict_context_vec(self, x_test, x_mi_test, keep_prob=1.0):
        x_test = tf.convert_to_tensor(x_test, dtype=tf.float32)
        x_mi_test = tf.convert_to_tensor(x_mi_test, dtype=tf.float32)
        self.forward(x_test, x_mi_test, training=False, keep_prob=keep_prob)
        return self.context_vec.numpy()

    def get_z_mean_and_std(self, x_test, x_mi_test, keep_prob=1.0):
        x_test = tf.convert_to_tensor(x_test, dtype=tf.float32)
        x_mi_test = tf.convert_to_tensor(x_mi_test, dtype=tf.float32)
        self.forward(x_test, x_mi_test, training=False, keep_prob=keep_prob)
        return self.z_mean.numpy(), self.z_std.numpy()