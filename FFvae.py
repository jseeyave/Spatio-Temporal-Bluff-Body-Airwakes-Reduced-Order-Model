import os
# ==========================================================================================================
os.environ["TF_GPU_ALLOCATOR"] = "cuda_malloc_async"
# Enable TF XLA JIt compilation - dynamically chose best-performing algorithm
os.environ['TF_XLA_FLAGS'] = '--tf_xla_auto_jit=2'
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2' 
import numpy as np
import tensorflow as tf
#import scipy.ndimage
import os
from tqdm import tqdm
from tensorflow.keras import layers, Model
#import matplotlib.pyplot as plt
import threading
import keras
import tensorflow.keras.backend as K
from Gradient3d import gradient3d
# =========================================================================================================
tf.config.optimizer.set_jit(True)
#tf.config.set_visible_devices([], 'GPU')                      # Disable GPU
gpus = tf.config.experimental.list_physical_devices('GPU')
for gpu in gpus:
    tf.config.experimental.set_memory_growth(gpu, True)
############################################################################################################
# Inputs ###################################################################################################
# Name = 'UNETpTemb'
val_split=0.2
initial_lr=0.0005
latent_dim = 128
BS = 16
act = 'tanh'
enc_in_shape = (176,80,64,3)
save_every_n_epochs = 50
num_timesteps=1050
epochs = 3000
cases = ['A0','A20','A30','A45']
# initial_epoch = 750
############################################################################################################
# DataLoader ###############################################################################################
class CFDDataLoader:
    def __init__(self,velocity_target_dir, cases, batch_size=4, num_timesteps=1000, val_split=0.2, shuffle=True, vel_shape=enc_in_shape):

        self.velocity_target_dir = velocity_target_dir  # Directory containing per-timestep files
        self.batch_size = batch_size
        self.num_timesteps = num_timesteps
        self.val_split = val_split
        self.shuffle = shuffle
        self.vel_shape = vel_shape
        self.cases = cases

        # Create list of (case_idx, time) pairs
        self.indices = [(c, t) for c in self.cases for t in range(self.num_timesteps)]

        if self.shuffle:
            np.random.shuffle(self.indices)

        # Split into train & validation sets
        split_idx = int(len(self.indices) * (1 - self.val_split))
        self.train_indices = self.indices[:split_idx]
        self.val_indices = self.indices[split_idx:]


    def load_sample(self, case_idx, t):
        # Construct file path dynamically
        velocity_file = os.path.join(self.velocity_target_dir, f"case_{case_idx}_time_{t}.npy")

        # Ensure file exists
        if not os.path.exists(velocity_file):
            raise FileNotFoundError(f"Missing velocity file: {velocity_file}")

        # Load velocity target file
        velocity_field = np.load(velocity_file)  # Load only required timestep
        
        return velocity_field

    def data_generator(self, indices):
        for case_idx, t in indices:
            yield self.load_sample(case_idx, t)

    def get_dataset(self, mode="train"):
        if mode == "train":
            dataset = tf.data.Dataset.from_generator(
                lambda: self.data_generator(self.train_indices),
                output_signature=(
                    tf.TensorSpec(shape=self.vel_shape, dtype=tf.float32)  # Velocity Field
                )
            )
        elif mode == "val":
            dataset = tf.data.Dataset.from_generator(
                lambda: self.data_generator(self.val_indices),
                output_signature=(                     
                    tf.TensorSpec(shape=self.vel_shape, dtype=tf.float32)  # Velocity Fiel
                )
            )

        # Shuffle and batch data
        dataset = dataset.shuffle(buffer_size=1000) if self.shuffle and mode == "train" else dataset
        dataset = dataset.batch(self.batch_size, drop_remainder=True)

        # Prefetch to hide I/O latency
        dataset = dataset.prefetch(tf.data.experimental.AUTOTUNE)

        return dataset

# Define paths
velocity_target_dir = "/home/jseeyave3/AngSweep10ms/UVWall.Normby3/"  # Folder containing `case_{case_idx}_timestep_{t}.npy` files
# Initialize Data Loader

data_loader = CFDDataLoader(
    velocity_target_dir=velocity_target_dir,  # Directory with per-timestep files
    cases=cases,
    batch_size=BS,
    num_timesteps=num_timesteps,
    val_split=val_split,
    vel_shape = enc_in_shape,
)    

# Asynchronous Model Saving Function
def save_model_async(model, saveM,saveW):
    def save():
        #model.save(saveM)
        model.save_weights(saveW)
        print(f"============Model saved at {saveM and saveW}.")
    threading.Thread(target=save).start()

# Custom Reduce LR on Plateau Function
def reduce_lr_on_plateau(optimizer, val_loss, best_val_loss, patience, factor=0.25, min_lr=1e-9):
    """Manually reduce learning rate when validation loss plateaus."""
    if val_loss >= best_val_loss:
        reduce_lr_on_plateau.counter += 1  # Track bad epochs
    else:
        reduce_lr_on_plateau.counter = 0  # Reset if loss improves

    if reduce_lr_on_plateau.counter >= patience:
        new_lr = max(float(optimizer.learning_rate.numpy()) * factor, min_lr)  # Ensure new_lr is float
        optimizer.learning_rate.assign(new_lr)  # Update LR correctly
        print(f"Reducing learning rate to {new_lr:.6f}")
        reduce_lr_on_plateau.counter = 0  # Reset counter
    
    return max(float(optimizer.learning_rate.numpy()), min_lr)  # Ensure return value is float

# Initialize counter as an attribute of the function
reduce_lr_on_plateau.counter = 0

def clip_logvar(x):
    return tf.clip_by_value(x, -10.0, 2.0)   

def cvae_encoder(x,latent_dim):
    x = layers.Conv3D(32, 3, strides=2, padding='same')(x)
    x = layers.LeakyReLU()(x)
    x = layers.Conv3D(128, 3, strides=2, padding='same')(x)
    x = layers.LeakyReLU()(x)
    x = layers.Conv3D(512, 3, strides=2, padding='same')(x)
    x = layers.LeakyReLU()(x)
    x = layers.Conv3D(2048, 3, strides=2, padding='same')(x)
    x = layers.LeakyReLU()(x)
    x = layers.Flatten()(x)
    x = layers.Dense(512)(x)
    x = layers.LeakyReLU()(x)

    mu = layers.Dense(latent_dim)(x)
    logvar = layers.Dense(latent_dim)(x)
    logvar = layers.Activation(clip_logvar)(logvar)
    return mu, logvar

def sample_latent(mu, logvar):
    """Reparameterization Trick as a Lambda Layer"""
    @keras.saving.register_keras_serializable()
    def reparameterization_trick(args):
        mu, logvar = args
        epsilon = tf.random.normal(shape=tf.shape(mu))
        return mu + tf.exp(0.5 * logvar) * epsilon

    return layers.Lambda(reparameterization_trick, name='latent')([mu, logvar])

def cvae_decoder(z):
    x = layers.Dense(11*5*4*2048)(z)
    x = layers.LeakyReLU()(x)
    x = layers.Reshape((11, 5, 4, 2048))(x)
    x = layers.Conv3DTranspose(512, 3, strides=2, padding='same')(x)
    x = layers.LeakyReLU()(x)
    x = layers.Conv3DTranspose(128, 3, strides=2, padding='same')(x)
    x = layers.LeakyReLU()(x)
    x = layers.Conv3DTranspose(32, 3, strides=2, padding='same')(x)
    x = layers.LeakyReLU()(x)
    x = layers.Conv3DTranspose(8, 3, strides=2, padding='same')(x)
    x = layers.LeakyReLU()(x)
    return layers.Conv3D(3, 3, activation='tanh', padding='same')(x)

    
def cvae_model(input_shape,  latent_dim):
    """CVAE Model"""
    x_input = tf.keras.Input(shape=input_shape)

    mu, logvar = cvae_encoder(x_input, latent_dim)
    z = sample_latent(mu, logvar)
    decoded = cvae_decoder(z)

    return tf.keras.Model(inputs=x_input, outputs=[decoded, mu, logvar])

def frequency_aware_loss(y_true, y_pred):
    # MSE loss
    mse_loss = tf.reduce_mean(tf.abs(y_true - y_pred))
    
    dx_t, dy_t, dz_t = gradient3d(y_true)
    dx_p, dy_p, dz_p = gradient3d(y_pred)
    
    # Gradient loss (preserves edges/high-frequency details)
    grad_loss = tf.reduce_mean((dx_t-dx_p)**2 + (dy_t-dy_p)**2 + (dz_t-dz_p)**2)
    
    return mse_loss,  grad_loss
    
# Loss Function
def cvae_loss(x, x_recon, mu, logvar,beta=0.01):
    l1, grad_loss = frequency_aware_loss(x, x_recon) 
    recon_loss = l1 + 10*grad_loss
    
    kl_loss = -0.5 * tf.reduce_mean(1 + logvar - tf.square(mu) - tf.exp(logvar))  # KL divergence
        
    return (1000*recon_loss) + (beta*kl_loss), recon_loss, kl_loss, tf.reduce_mean(tf.exp(logvar))

optimizer = tf.keras.optimizers.Adam(learning_rate=initial_lr)
clip_value = 50.0 
@tf.function
def train_step(velocity):
    with tf.GradientTape() as tape:
        x_recon, mu, logvar = model(velocity)
        loss, mse, kll, sig = cvae_loss(velocity, x_recon, mu, logvar)
    gradients = tape.gradient(loss, model.trainable_variables)
    gradients, _ = tf.clip_by_global_norm(gradients, clip_value)  # Clip gradients
    optimizer.apply_gradients(zip(gradients, model.trainable_variables))
    return mu, logvar, loss, mse, kll, sig

@tf.function
def val_step(velocity):
    x_recon, mu, logvar = model(velocity)
    loss, mse, kll, sig = cvae_loss(velocity, x_recon, mu, logvar)    
    return mu, logvar, loss, mse, kll, sig
    
###############################################################################################################
###############################################################################################################
####################################### Training Loop
def train_cvae(cvae, train_dataset, val_dataset,epochs=2000, patience=20, factor=0.25, min_lr=1e-9, saveM="cvae_model.keras",saveW="cvae.weights.h5", stop=50):
    ecount = eqep = ered = early_stop_counter = 0
    nlr = initial_lr
    best_val_loss = float('inf')  # Track best validation loss
    num_Tbatches = len(data_loader.train_indices) // data_loader.batch_size
    num_Vbatches = len(data_loader.val_indices) // data_loader.batch_size
    
    for epoch in range(epochs):
        
        ###############################################################################################################
        ####################################### Training Phase
        train_loss = tmse = tkll = tmu_m = tmu_n = tlg_m = tlg_n = tsig = 0
        print(f"\nEpoch {epoch + 1}/{epochs}: Training")
        with tqdm(total=num_Tbatches, desc="Progress", unit="batch") as pbar:
            for velocity in train_dataset:
                mu, logvar, loss, mse, kll, sig = train_step(velocity)
                train_loss += loss.numpy()
                tmse += mse.numpy()
                tkll += kll.numpy()
                tmu_n += tf.norm(tf.reduce_mean(mu, axis=0)).numpy()
                tmu_m += tf.reduce_mean(mu).numpy()
                tlg_n += tf.norm(tf.reduce_mean(logvar, axis=0)).numpy()
                tlg_m += tf.reduce_mean(logvar).numpy()
                tsig += sig.numpy()
                pbar.update(1)  # Update progress bar
        ###############################################################################################################
        ####################################### Validation Phase
        val_loss = vmse = vkll = vmu_m = vmu_n = vlg_m = vlg_n = vsig= 0
        for velocity in val_dataset:        
            mu, logvar, loss, mse, kll, sig = val_step(velocity)
            val_loss += loss.numpy()
            vmse += mse.numpy()
            vkll += kll.numpy()
            vmu_n += tf.norm(tf.reduce_mean(mu, axis=0)).numpy()
            vmu_m += tf.reduce_mean(mu).numpy()
            vlg_n += tf.norm(tf.reduce_mean(logvar, axis=0)).numpy()
            vlg_m += tf.reduce_mean(logvar).numpy()
            vsig += sig.numpy()

        ###############################################################################################################
        ####################################### Compute Average Losses        
        train_loss /= num_Tbatches
        tmse /= num_Tbatches
        tkll /= num_Tbatches
        tmu_n /= num_Tbatches
        tmu_m /= num_Tbatches
        tlg_n /= num_Tbatches
        tlg_m /= num_Tbatches
        tsig /= num_Tbatches

        val_loss /= num_Vbatches        
        vmse /= num_Vbatches
        vkll /= num_Vbatches
        vmu_n /= num_Vbatches
        vmu_m /= num_Vbatches
        vlg_n /= num_Vbatches
        vlg_m /= num_Vbatches
        vsig /= num_Vbatches
        
        print(f"Epoch {epoch + 1}, Tmae: {tmse:.4f}, Tkll: {tkll:.2f}, T Loss: {train_loss:.2f}, Tsig: {tsig:.2f}, Tmum:{tmu_m:.3f}, Tlgm:{tlg_m:.3f}, Tmun:{tmu_n:.3f}, Tlgn:{tlg_n:.3f}") 
        print(f"********, Vmae: {vmse:.4f}, Vkll: {vkll:.2f}, V Loss: {val_loss:.2f}, Vsig: {vsig:.2f}, Vmum:{vmu_m:.3f}, Vlgm:{vlg_m:.3f}, Vmun:{vmu_n:.3f}, Vlgn:{vlg_n:.3f}") 
        val_imp = best_val_loss-val_loss
        print(f"********, Best V Loss: {best_val_loss:.4f}, NI: {early_stop_counter:.1f}, LR: {nlr:.2E}") 
        
        nlr = reduce_lr_on_plateau(optimizer, vmse, best_val_loss, patience, factor, min_lr)             

        ###############################################################################################################
        ####################################### Early Stopping Logic
        if vmse < best_val_loss: # and vkll > 0.01:
            best_val_loss = vmse
            early_stop_counter = 0  # Reset counter
            save_model_async(cvae, saveM,saveW)  # Save best model asynchronously
        else:
            early_stop_counter += 1
            if early_stop_counter >= stop:
                print(f"Early stopping at epoch {epoch + 1} due to no improvement.")
                break  # Stop training            

if __name__ == '__main__':
    model = cvae_model(enc_in_shape, latent_dim)
    model.summary()
    '''
    try:
        model.load_weights('cvae.weights.h5')
        print('*********weight loaded')
    except ValueError as e:
        print(f"Error: {e}")
        raise  # Stop execution by re-raising the exception
    #'''
    # Create train & validation datasets
    train_dataset = data_loader.get_dataset(mode="train")
    val_dataset = data_loader.get_dataset(mode="val")
    train_cvae(model, train_dataset, val_dataset,epochs=2000, patience=25, factor=0.4, min_lr=1e-8, saveM="cvae.keras",saveW="cvae.weights.h5", stop=200)
