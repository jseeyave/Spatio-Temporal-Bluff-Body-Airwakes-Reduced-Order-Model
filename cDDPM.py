import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'  # 0 = all logs, 1 = warnings + errors, 2 = errors only, 3 = fatal only
# ==========================================================================================================
import math, functools, tensorflow as tf
from tensorflow.keras import layers, Model, models
from dataclasses import dataclass
from tensorflow.keras import mixed_precision as mp
import numpy as np
import threading
from tqdm import tqdm
from CBAM import CBAM
# ==========================================================================================================
os.environ["TF_GPU_ALLOCATOR"] = "cuda_malloc_async"
os.environ['TF_XLA_FLAGS'] = '--tf_xla_auto_jit=2'
# =======================================================================================================
tf.config.optimizer.set_jit(True)
gpus = tf.config.experimental.list_physical_devices('GPU')
for gpu in gpus:
    tf.config.experimental.set_memory_growth(gpu, True)
############################################################################################################
# Constants
############################################################################################################
BS= 32
LR = 1e-5
epochs = 500
GN = 8
Xshape = (176,80,64,3)
cases = ['A0','A20','A30','A45']
num_timesteps=1050
############################################################################################################
# DataLoader ###############################################################################################
class CFDDataLoader:
    def __init__(self, velocity_target_dir, blury_dir,cases, batch_size=4, num_timesteps=10000, shuffle=True, vel_shape=(176,80,64,3)):       
        self.velocity_target_dir = velocity_target_dir
        self.blury_dir = blury_dir
        self.batch_size = batch_size
        self.num_timesteps = num_timesteps
        self.shuffle = shuffle
        self.vel_shape = vel_shape
        self.cases = cases

        self.indices = [(c, t) for c in self.cases for t in range(self.num_timesteps)]        
        np.random.shuffle(self.indices)

        # Split into train & validation sets
        split_idx = int(len(self.indices) * (1 - 0.1))
        self.train_indices = self.indices[:split_idx]
        self.val_indices = self.indices[split_idx:]

    def load_sample(self, case_idx,t):
        velocity_file = os.path.join(self.velocity_target_dir, f"case_{case_idx}_time_{t}.npy")
        blur_file = os.path.join(self.blury_dir, f"vaepred.case_{case_idx}_time_{t}.npy")
        if not os.path.exists(velocity_file):
            raise FileNotFoundError(f"Missing velocity file: {velocity_file}")
        if not os.path.exists(blur_file):
            raise FileNotFoundError(f"Missing velocity file: {blur_file}")
            
        velocity_field = np.load(velocity_file)
        blurry_field = np.load(blur_file)/3 
        
        return velocity_field, blurry_field

    def data_generator(self, indices):
        for case_idx, t in indices:
            yield self.load_sample(case_idx, t)

    def get_dataset(self, mode="train"):
        if mode == "train":
            dataset = tf.data.Dataset.from_generator(
                lambda: self.data_generator(self.train_indices),
                output_signature=(
                    tf.TensorSpec(shape=self.vel_shape, dtype=tf.float32),                     
                    tf.TensorSpec(shape=self.vel_shape, dtype=tf.float32),  
                )
            )
        elif mode == "val":
            dataset = tf.data.Dataset.from_generator(
                lambda: self.data_generator(self.val_indices),
                output_signature=(
                    tf.TensorSpec(shape=self.vel_shape, dtype=tf.float32), 
                    tf.TensorSpec(shape=self.vel_shape, dtype=tf.float32), 
                )
            )
        dataset = dataset.shuffle(buffer_size=500) if self.shuffle else dataset 
        dataset = dataset.batch(self.batch_size, drop_remainder=True)
        dataset = dataset.prefetch(tf.data.experimental.AUTOTUNE)
        return dataset

# Define paths
#velocity_target_dir =  "/home/jseeyave3/AngSweep10ms/UVWall.Normby3/"  
#blury_dir = "/home/jseeyave3/A.erf/ffvae/predictions/"   
velocity_target_dir = "/data/ae-jral/jseeyave3/UVWall.Normby3/"
blury_dir = "/data/ae-jral/jseeyave3/ffvae/predictions/"   # Folder containing `case_{case_idx}_timestep_{t}.npy` files


# Initialize Data Loader
data_loader = CFDDataLoader(
    velocity_target_dir=velocity_target_dir,  # Directory with per-timestep files
    blury_dir = blury_dir,
    cases=cases,
    batch_size=BS,
    num_timesteps=num_timesteps,
    vel_shape = Xshape,
)   

def save_model_async(model, saveM,saveW):
    def save():
        # model.save(saveM)
        model.save_weights(saveW)
        print(f"================ Model saved at {saveM and saveW} ================ ")
    threading.Thread(target=save).start()
#########################################################################################################################################
#########################################################################################################################################
class FiLM(layers.Layer):
    def __init__(self, num_channels, cond_mode='vector', **kwargs):
        super(FiLM, self).__init__(**kwargs)
        self.num_channels = num_channels
        assert cond_mode in ['vector', 'map'], "cond_mode must be 'vector' or 'map'"
        self.cond_mode = cond_mode
        if cond_mode == 'vector':
            self.gamma_gen = layers.Dense(num_channels)
            self.beta_gen  = layers.Dense(num_channels)
        else:
            self.gamma_conv = layers.Conv3D(num_channels, kernel_size=1)
            self.beta_conv  = layers.Conv3D(num_channels, kernel_size=1)

    def call(self, feature_map, cond ):
        if self.cond_mode == 'vector':
            # vector conditioning
            gamma = self.gamma_gen(cond)
            beta  = self.beta_gen(cond)
            gamma = layers.Reshape((1, 1, 1, self.num_channels))(gamma)
            beta  = layers.Reshape((1, 1, 1, self.num_channels))(beta)
        else:
            # spatial conditioning
            gamma = self.gamma_conv(cond)
            beta  = self.beta_conv(cond)
        return feature_map * gamma + beta

class TimeEmbedding(layers.Layer):
    def __init__(self, dim):
        super().__init__()
        self.dim = dim
        self.proj = layers.Dense(dim)
    
    def build(self, input_shape):
        self.built = True
        
    def call(self, t):
        t = tf.cast(t, tf.float32)
        half_dim = self.dim // 2
        emb = tf.math.log(10000.0) / (half_dim - 1)
        emb = tf.exp(tf.range(half_dim, dtype=tf.float32) * -emb)
        emb = t[:, None] * emb[None, :]
        emb = tf.concat([tf.sin(emb), tf.cos(emb)], axis=-1)
        return emb 
        
class EfficientMultiHeadAttention(layers.Layer):
    def __init__(self, d_model, num_heads, **kwargs):
        super().__init__(**kwargs)
        assert d_model % num_heads == 0, "d_model must be divisible by num_heads"
        self.num_heads = num_heads
        self.depth = d_model // num_heads
        
        self.wq = layers.Dense(d_model)
        self.wk = layers.Dense(d_model)
        self.wv = layers.Dense(d_model)
        
        self.ln1 = tf.keras.layers.LayerNormalization()
        self.ff = tf.keras.Sequential([
            tf.keras.layers.Dense(d_model * 4, activation='gelu'),
            tf.keras.layers.Dense(d_model)
        ])
        self.ln2 = tf.keras.layers.LayerNormalization()  

    def split_heads(self, x):
        batch_size = tf.shape(x)[0]
        x = tf.reshape(x, (batch_size, -1, self.num_heads, self.depth))
        return tf.transpose(x, perm=[0, 2, 1, 3])

    def call(self, q, kv):
        B,H,W,D,C = q.shape
        q = layers.Reshape((-1, C))(q)
        inq = q
        
        kv = layers.Reshape((-1, C))(kv)
        
        q = self.split_heads(self.wq(q))  # (b, h, n, d_k)
        k = self.split_heads(self.wk(kv))  # (b, h, n, d_k)
        v = self.split_heads(self.wv(kv))  # (b, h, n, d_v)

        k_norm = tf.nn.softmax(tf.transpose(k, [0,1,3,2]), axis=-1)  # (b, h, d_k, n) 
        q_norm = tf.nn.softmax(q, axis=-1)                           # (b, h, n, d_k)

        context = tf.matmul(k_norm, v)  # (b, h, d_k, depth_v)

        attn = tf.matmul(q_norm, context)  # (b, h, n, depth_v)

        attn = tf.transpose(attn, [0, 2, 1, 3])  # (b, n, h, depth_v)
        batch_size = tf.shape(attn)[0]
        seq_len = tf.shape(attn)[1]
        attn = tf.reshape(attn, [batch_size, seq_len, C])  # (b, n, d_model)

        attention = self.ln1(inq + attn)
        out = self.ln2(attention + self.ff(attention))
        output = layers.Reshape((H, W, D, C))(out)
        return output

class mha(tf.keras.layers.Layer):
    """Self-attention block for the UNet"""
    def __init__(self, channels, num_heads=16,dim_head=128):
        super().__init__()
        self.channels = channels
        self.num_heads = num_heads
        self.mha = tf.keras.layers.MultiHeadAttention(
            num_heads=num_heads,
            key_dim=channels//num_heads,
            value_dim=channels//num_heads)

        self.ln1 = tf.keras.layers.LayerNormalization()
        self.ff = tf.keras.Sequential([
            tf.keras.layers.Dense(channels * 4, activation='gelu'),
            tf.keras.layers.Dense(channels)])
        self.ln2 = tf.keras.layers.LayerNormalization()
    
    def call(self, x, training=False):
        B, H, W, D, C = x.shape #tf.shape(x)[0], tf.shape(x)[1], tf.shape(x)[2], tf.shape(x)[3], tf.shape(x)[4]
               
        x_flat = layers.Reshape((-1, C))(x)
               
        attention_output = self.mha(x_flat, x_flat, training=training)
        x_flat = self.ln1(x_flat + attention_output)
               
        x_flat = self.ln2(x_flat + self.ff(x_flat))
        
        xx = layers.Reshape((H, W, D, C))(x_flat)
        
        return xx          
    
def ResidualBlock4D(x, filters, t_d, cond):
    shortcut = x
    if x.shape[-1] != filters:
        shortcut = layers.Conv3D(filters, 1, padding="same")(shortcut)

    h = layers.GroupNormalization(groups=GN, epsilon=1e-5)(x)
    h = layers.Activation("swish")(h)   
    h = layers.Conv3D(filters, 3, padding="same")(h)
		
    h = FiLM(filters, cond_mode='vector')(h, t_d) 
    h = FiLM(filters, cond_mode='map')(h, cond) 

    h = layers.GroupNormalization(groups=GN, epsilon=1e-5)(h)
    h = layers.Activation("swish")(h)         
    h = layers.Conv3D(filters, 3, padding="same")(h)

    return layers.Add()([shortcut, h])   


def Downsample4D(x, filters):
    return layers.Conv3D(filters, 3, strides=2, padding="same")(x)


def Upsample4D(x, filters):
    x = layers.UpSampling3D(2)(x)
    return layers.Conv3D(filters, 3, strides=1, padding="same")(x)  

             
def UNet4D(input_shape, dim=8, NB=4):
    xi = layers.Input(shape=input_shape)  # Shape: (height, width, depth, time_steps, channels)
    t_d = layers.Input(shape=())
    condi = layers.Input(shape=input_shape)
    
    # Time embedding for diffusion timestep
    t_de = TimeEmbedding(dim)(t_d)
    t_de = layers.Dense(dim*4,activation='swish')(t_de)
    t_de = layers.Dense(dim*4)(t_de)

    cond = layers.Conv3D(16, 3, padding="same",activation='swish')(condi)
    cond = layers.Conv3D(32, 3, padding="same",activation='swish')(cond)    
    cond = CBAM()(cond)
    cond = layers.Conv3D(32, 3, padding="same")(cond)
    # Encoder ==========================================================
    x = layers.Conv3D(dim, 3, padding="same",name='init_conv')(xi)    
    
    skips = []
    for i in range(NB):
        x = ResidualBlock4D(x, dim*(2**(i*2)), t_de, cond)
        x = ResidualBlock4D(x, dim*(2**(i*2)), t_de, cond)
        if i == 2:
            x = EfficientMultiHeadAttention(dim*(2**(i*2)),4)(x,x)
        skips.append(x)
        x = Downsample4D(x, dim*(2**(i*2)))
        cond = Downsample4D(cond, dim*(2**(i*2)))
    
    # Bottleneck ==========================================================
    x = ResidualBlock4D(x, 4*dim*(2**NB), t_de, cond)
    xb = mha(4*dim*(2**NB), num_heads=4, dim_head=4*dim*(2**NB))(x, training=True)
    #xb = EfficientMultiHeadAttention(dim*(2**NB),4)(x,x)
    x = ResidualBlock4D(xb, 4*dim*(2**NB), t_de, cond)
    # Decoder ==========================================================         

    for i in reversed(range(NB)):
        x = Upsample4D(x, dim*(2**(i*2)))
        cond = Upsample4D(cond, dim*(2**(i*2)))
        x = layers.Concatenate()([x, skips[i]])
        x = ResidualBlock4D(x, dim*(2**(i*2)), t_de, cond)
        x = ResidualBlock4D(x, dim*(2**(i*2)), t_de, cond)
        if i == 2:
            x = EfficientMultiHeadAttention(dim*(2**(i*2)),4)(x,x)
        
    # Output ==========================================================
    x = layers.GroupNormalization(groups=GN, epsilon=1e-5)(x)
    x = layers.Activation("swish")(x)   
    out = layers.Conv3D(input_shape[-1], 3, padding="same")(x)
    return Model(inputs=[xi, t_d, condi], outputs=out)       
#########################################################################################################################################
#########################################################################################################################################
# -------------------------------------------------
# 2.  Configuration
# -------------------------------------------------
@dataclass
class DiffusionCfg:
    num_steps: int = 1000          # training DDPM steps (teacher)
    img_size: int = 256
    channels: int = 1
    snr_gamma: float = 5.0         # loss re-weighting (set =0 for plain MSE)
    mixed_precision: bool = False

# -------------------------------------------------
# 1.  Noise schedule and helper functions
# -------------------------------------------------
def alpha_sigma(t: tf.Tensor):
    half_pi = tf.constant(math.pi / 2, t.dtype)
    return tf.cos(t * half_pi), tf.sin(t * half_pi)

def _expand(x):    # shapes (B,) -> (B,1,1,1,1)
    return x[:, None, None, None, None]
    
def v_from(x0, eps, t):
    """Velocity v = alpha_t·ε − sigma_t·x₀."""
    alpha, sigma = alpha_sigma(t)    
    return _expand(alpha) * eps - _expand(sigma) * x0

def v2x0_eps(x_t, v, t):
    """Recover x₀ and ε from (x_t, v)."""
    alpha, sigma = alpha_sigma(t)
    denom = alpha**2 + sigma**2 
    alpha = _expand(alpha)
    sigma = _expand(sigma)
    x_t = tf.cast(x_t,   tf.float32)
    v = tf.cast(v,   tf.float32)
    x0 = (alpha * x_t - sigma * v) / _expand(denom)
    eps = (alpha * v + sigma * x_t) / _expand(denom)
    return x0, eps

# -------------------------------------------------
# 3.  Core Diffusion class
# -------------------------------------------------
class Diffusion(tf.keras.Model):
    def __init__(self, unet, cfg: DiffusionCfg):
        super().__init__()
        self.unet, self.cfg = unet, cfg

        for blk in self.unet.layers:                        
            if isinstance(blk, tf.keras.Model):             
                blk.call = tf.function(
                    blk.call,
                    jit_compile=True,
                    autograph=False,)         
        _ = self.unet([tf.zeros((1,*Xshape)),tf.zeros((1,)),tf.zeros((1,*Xshape))])

        self.t_schedule = tf.linspace(1.0 / cfg.num_steps,
                                      1.0,
                                      cfg.num_steps)
    # -------------------------------------------------
    # 3.1 Forward / training-time loss
    # -------------------------------------------------
    @tf.function()
    def loss_fn(self, x0, blurry):
        """Single diffusion timestep sampled uniformly."""
        B = tf.shape(x0)[0]
        t_idx = tf.random.uniform([B], 1, self.cfg.num_steps-1,dtype=tf.int32)
        t = tf.gather(self.t_schedule, t_idx)        

        eps = tf.random.normal(tf.shape(x0), dtype=x0.dtype)
        alpha, sigma = alpha_sigma(t)
        x_t = _expand(alpha) * x0 + _expand(sigma) * eps

        v_target = v_from(x0, eps, t)

        v_pred = self.unet([x_t, t, blurry], training=True)  # assumes UNet takes (x_t, timestep)

        # ------ MSE with optional SNR-based weight ------
        v_target=tf.cast(v_target, v_pred.dtype)
        mse = tf.reduce_mean(tf.square(v_pred - v_target), axis=[1, 2, 3, 4])
        if self.cfg.snr_gamma > 0:
            snr = (alpha ** 2) / tf.maximum(sigma ** 2, 1e-8)
            snr  = tf.cast(snr, tf.float32)
            snr  = tf.clip_by_value(snr, 0.0, 1e12)
            gamma = tf.constant(self.cfg.snr_gamma, tf.float32)
            weights32 = tf.minimum(gamma, snr)/(snr+1.0)
            tf.debugging.assert_less_equal(tf.reduce_max(weights32), 10.0,"weights32 is unexpectedly large")  
            weights = tf.cast(weights32, mse.dtype)
            mse *= weights
        if self.cfg.snr_gamma == 0:
            weights32 = 0
        loss = tf.reduce_mean(mse)
        rw = tf.reduce_mean(weights32)
        nt = tf.norm(tf.cast(v_target, tf.float32))
        np = tf.norm( tf.cast(v_pred, tf.float32)) 
        return loss, rw, nt, np

    # -------------------------------------------------
    # 3.2 DDIM / probability-flow sampler
    # -------------------------------------------------
    @tf.function()
    def ddim_sample_cond(self,
                        blurry,        # (B,H,W,C)  clean condition
                        steps=50,
                        eta=0.0):      # 0 = deterministic probability-flow

        B, H, W, D, C = tf.unstack(tf.shape(blurry))
        res_t = tf.random.normal((B, H, W, D, C), dtype=blurry.dtype)

        # --- timetable --------------------------------------------------------
        t_seq = tf.linspace(1.0, 1.0 / self.cfg.num_steps, steps)
        # t_seq = tf.linspace(1.0, 0.0, steps+1)[:-1]

        for i in tf.range(steps):
            t      = t_seq[i]                                # scalar float32
            t_b    = tf.fill([B], t)

            # 1.  Velocity prediction            
            v_pred = self.unet([res_t, t_b, blurry], training=False)

            # 2.  Recover epŝ & x0̂  (only needed for DDIM-step branch)            
            x0_hat, eps_hat  = v2x0_eps(res_t, v_pred, t_b)

            # 3.  Choose previous timestep (t_prev = 0 for last iteration)
            if i == steps - 1:
                res_t = x0_hat            # x_{t=0}
            else:
                t_prev         = t_seq[i+1]
                t_prev_b       = tf.fill([B], t_prev)
                alpha_prev, sigma_prev = alpha_sigma(t_prev_b)
                if eta == 0.0:                    # deterministic DDIM (PF-ODE)
                    res_t = (_expand(alpha_prev) * x0_hat +
                            _expand(sigma_prev) * eps_hat)
                else:                             # stochastic DDIM
                    noise      = tf.random.normal(tf.shape(res_t), dtype=res_t.dtype)
                    eps_prime  = ( tf.sqrt(1.0 - eta**2) * eps_hat + eta * noise )      # DDIM eq. 11
                    res_t      = ( _expand(alpha_prev) * x0_hat + _expand(sigma_prev) * eps_prime )
        return res_t

log_dir = "./logs/grad_norm_test"          # one run = one folder
writer  = tf.summary.create_file_writer(log_dir)
    
@tf.function()
def train_step(r0,blurry):
    with tf.GradientTape() as tape:
        loss, wsnr, tnorm, pnorm = diffusion.loss_fn(r0,blurry)
        loss = loss
    grads = tape.gradient(loss, diffusion.trainable_weights)        
    opt.apply_gradients(zip(grads, diffusion.trainable_weights))  
    
    return loss, wsnr, tnorm, pnorm, grads

@tf.function()
def val_step(r0,blurry):
    loss, *_ = diffusion.loss_fn(r0,blurry)    
    return loss

opt = tf.keras.optimizers.Adam(learning_rate=1e-5)

unet = UNet4D(Xshape)           
# unet.summary()
cfg = DiffusionCfg(num_steps=1000)
diffusion = Diffusion(unet, cfg)

'''
_ = unet([tf.zeros((1,*Xshape)),tf.zeros((1,)),tf.zeros((1,*Xshape))])
try:
    unet.load_weights('vCDM2.25.snr5.btw.weights.h5')
    print('*********weight loaded')
except ValueError as e:
    print(f"Error: {e}")
    raise  # Stop execution by re-raising the exception #'''
    
#dataset = data_loader.get_dataset(mode="train")   # prepare dataset of [-1,1] images
train_dataset = data_loader.get_dataset(mode="train")
val_dataset = data_loader.get_dataset(mode="val")
num_Tbatches = len(data_loader.train_indices) // data_loader.batch_size
print("steps/epoch:", num_Tbatches) 
best_val_loss= best_t_loss = float('inf')
wait = twait =0
global_step = 0
for epoch in range(epochs):
    epoch_train_loss = []
    epoch_val_loss = []
    twsnr = []
    ttnorm = []
    tpnorm = []
    writer.flush() 
    for sharp, blurry in tqdm(train_dataset, total=num_Tbatches, desc="Progress", unit="batch"): 
        r0 = (sharp - blurry)*25       
        loss, wsnr, tnorm, pnorm, gnorm = train_step(r0,blurry)
        gnorm = tf.linalg.global_norm(gnorm)
        epoch_train_loss.append(loss.numpy())
        twsnr.append(wsnr.numpy())
        ttnorm.append(tnorm.numpy())
        tpnorm.append(pnorm.numpy())        
    tloss = np.mean(epoch_train_loss)
    tw = np.mean(twsnr)
    ttn = np.mean(ttnorm)
    tpn = np.mean(tpnorm)
    gap = ttn-tpn

    for sharp, blurry in val_dataset:                       # x_batch ∈ [−1,1]
        r0 = (sharp - blurry)*25
        loss = val_step(r0,blurry)
        epoch_val_loss.append(loss.numpy())        
    vloss = np.mean(epoch_val_loss)
    print(f"Epoch{epoch:3d} | TL= {tloss:.3E} | VL= {vloss:.3E} | tw= {tw:.2f} | tn= {ttn:.2E} | pn= {tpn:.2E} | gap= {gap:.2E}")
    print(f"******** BTL= {best_t_loss:.3E} | BVL= {best_val_loss:.3E} | tNI= {twait:.3f} | vNI= {wait:.3f}")

    if (epoch+1)%50==0:
        modelName = f"CDM.effattn2_model.m11.epoch{epoch + 1}.keras"
        weightsname = f"vCDM2.25.epoch{epoch + 1}.weights.h5"
        save_model_async(unet, saveM=modelName ,saveW=weightsname)
    if vloss < best_val_loss:
        best_val_loss = vloss
        wait = 0        
        save_model_async(unet, saveM="CDM.model.keras",saveW="vCDM2.25.snr5.bvw.weights.h5")
    else:
        wait += 1
    if tloss < best_t_loss:
        best_t_loss = tloss
        twait = 0        
        save_model_async(unet, saveM="CDM.model.keras",saveW="vCDM2.25.snr5.btw.weights.h5")
    else:
        twait += 1


