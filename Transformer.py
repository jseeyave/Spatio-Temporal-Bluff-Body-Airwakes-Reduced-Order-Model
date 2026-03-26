import tensorflow as tf
from tensorflow.keras import layers, Model
import numpy as np
import os
import threading
from keras.callbacks import ModelCheckpoint, EarlyStopping, ReduceLROnPlateau, CSVLogger
from tqdm import tqdm
# ==========================================================================================================
os.environ["TF_GPU_ALLOCATOR"] = "cuda_malloc_async"
# Enable TF XLA JIt compilation - dynamically chose best-performing algorithm
os.environ['TF_XLA_FLAGS'] = '--tf_xla_auto_jit=0'
# =========================================================================================================
tf.config.optimizer.set_jit(True)
#tf.config.set_visible_devices([], 'GPU')                      # Disable GPU
#tf.keras.mixed_precision.set_global_policy('mixed_float16')   # Use float16
############################################################################################################
Name = 'TransEnc'
val_split=0.2
BS = 64
act = 'tanh'
latent_dim = 128
filters = [16,64,256,1024]
mask_shape = int(128)
enc_in_shape = (176,80,64,1)
save_every_n_epochs = 50
num_timesteps=1050
epochs = 3000
# Transformer
sequence_length = 100
d_model = 1024
num_heads = 8
dff = 256
latent_dim = 128
Nlayers = 6
############################################################################################################
# DataLoader ###############################################################################################
class CFDDataLoader:
    def __init__(self, ld_mask_file, target_dir, batch_size=4, num_timesteps=1000, val_split=0.2, sequence_length=10, shuffle=True, target_shape=latent_dim, mask_shape=mask_shape, d_model=d_model):

        self.ld_mask_file = ld_mask_file 
        self.target_dir = target_dir  # Directory containing per-timestep files
        self.batch_size = batch_size
        self.num_timesteps = num_timesteps
        self.val_split = val_split
        self.shuffle = shuffle
        self.target_shape = target_shape
        self.mask_shape = mask_shape
        self.d_model = d_model
        self.sequence_length = sequence_length

        self.ld_masks = np.load(ld_mask_file)  # Shape: (num_cases, X, Y, W, 1)
        self.num_cases = self.ld_masks.shape[0]
        
        self.target = np.load(target_dir)  # Load only required timestep
        
        self.indices = [(c, t) for c in range(self.num_cases) for t in range(self.num_timesteps- sequence_length + 1)]

        if self.shuffle:
            np.random.shuffle(self.indices)

        # Split into train & validation sets
        split_idx = int(len(self.indices) * (1 - self.val_split))
        self.train_indices = self.indices[:split_idx]
        self.val_indices = self.indices[split_idx:]

        
    def load_sample(self, case_idx, start_t):
        """Loads a single (mask, time) sample and corresponding velocity target from file."""
        target = self.target[case_idx, start_t:start_t + self.sequence_length,:]  # Load only required timestep
        mask = self.ld_masks[case_idx]
        tseq = np.array([start_t], dtype=np.float32) 
        
        return (mask, tseq), target

    def data_generator(self, indices):
        """Creates a generator for tf.data.Dataset."""
        for case_idx, start_t in indices:           
            yield self.load_sample(case_idx, start_t)


    def get_dataset(self, mode="train"):
        """Creates a tf.data.Dataset with prefetching for faster training."""
        if mode == "train":
            dataset = tf.data.Dataset.from_generator(
                lambda: self.data_generator(self.train_indices),
                output_signature=(
                    (tf.TensorSpec(shape=(self.mask_shape,), dtype=tf.float32),  # latent Mask
                     tf.TensorSpec(shape=(1,), dtype=tf.float32)),  # Time Step
                    tf.TensorSpec(shape=(self.sequence_length,self.target_shape), dtype=tf.float32)  # latent target
                )
            )
        elif mode == "val":
            dataset = tf.data.Dataset.from_generator(
                lambda: self.data_generator(self.val_indices),
                output_signature=(
                    (tf.TensorSpec(shape=(self.mask_shape,), dtype=tf.float32),  # latent Mask
                     tf.TensorSpec(shape=(1,), dtype=tf.float32)),  # Time Step
                    tf.TensorSpec(shape=(self.sequence_length,self.target_shape), dtype=tf.float32)  # latent target
                )
            )

        dataset = dataset.batch(self.batch_size)

        # Prefetch to hide I/O latency
        dataset = dataset.prefetch(tf.data.experimental.AUTOTUNE)

        return dataset

# Define paths
ld_mask_file = "/home/jseeyave3/FluidMasks/VAE/FMLatent.d128.npy"  # Contains all case masks
target_dir = "/home/jseeyave3/A.erf/ffvae/trainlatent_space128.npy"
# Initialize Data Loader
batch_size = BS
data_loader = CFDDataLoader(
    ld_mask_file=ld_mask_file,    
    target_dir=target_dir,  # Directory with per-timestep files
    batch_size=batch_size,
    num_timesteps=num_timesteps,
    val_split=val_split,
    sequence_length=sequence_length,
    shuffle=True,
    target_shape = latent_dim,
    mask_shape = mask_shape,
    d_model = d_model,
)   


# Asynchronous Model Saving Function
def save_model_async(model, saveM,saveW):
    def save():
        #model.save(saveM)
        model.save_weights(saveW)
        print(f"====================Model saved at {saveM and saveW}.====================")
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

def custom_cosine_similarity_loss(y_true, y_pred):
    y_true = tf.math.l2_normalize(y_true, axis=-1)  # Normalize true vectors
    y_pred = tf.math.l2_normalize(y_pred, axis=-1)  # Normalize predicted vectors
    cosine_similarity = tf.reduce_sum(y_true * y_pred, axis=-1)  # Compute dot product
    return 1 - cosine_similarity

# --------------------------
# 1. Positional Encoding Layer
# --------------------------
def get_angles(pos, i, d_model):
    i = tf.cast(i, tf.float32)
    angle_rates = 1 / tf.pow(10000.0, (2 * (i // 2)) / tf.cast(d_model, tf.float32))
    return tf.cast(pos, tf.float32)[..., tf.newaxis] * angle_rates


def positional_encoding_from_indices(pos, d_model):
    angle_rads = get_angles(pos, tf.range(d_model), d_model)
    angle_rads = tf.cast(angle_rads, tf.float32)
    angle_rads_even = tf.math.sin(angle_rads[..., 0::2])
    angle_rads_odd = tf.math.cos(angle_rads[..., 1::2])

    angle_rads_full = tf.concat([angle_rads_even[..., tf.newaxis], angle_rads_odd[..., tf.newaxis]], axis=-1)
    angle_rads_full = tf.reshape(angle_rads_full, tf.shape(angle_rads))
    return angle_rads_full

# --------------------------
# 2. Transformer Encoder Block
# --------------------------
class TransformerEncoderBlock(layers.Layer):
    def __init__(self, d_model, num_heads, dff, dropout_rate=0.1):
        super().__init__()
        self.mha = layers.MultiHeadAttention(num_heads=num_heads, key_dim=d_model//num_heads)
        self.ffn = tf.keras.Sequential([
            layers.Dense(d_model*4, activation='relu'),
            layers.Dense(d_model)
        ])
        self.layernorm1 = layers.LayerNormalization(epsilon=1e-6)
        self.layernorm2 = layers.LayerNormalization(epsilon=1e-6)
        self.dropout1 = layers.Dropout(dropout_rate)
        self.dropout2 = layers.Dropout(dropout_rate)

    def call(self, x):
        attn_output = self.mha(query=x, value=x, key=x)
        attn_output = self.dropout1(attn_output)
        x = self.layernorm1(x + attn_output)

        ffn_output = self.ffn(x)
        ffn_output = self.dropout2(ffn_output)
        return self.layernorm2(x + ffn_output)

# --------------------------
# 3. Main VAE Model
# --------------------------
class TransformerVAE(Model):
    def __init__(self,  d_model, num_heads, Nlayers):
        super().__init__()
        self.d_model = d_model        
        self.input_proj = tf.keras.layers.Dense(d_model)        
        self.transformer_blocks = [
            TransformerEncoderBlock(d_model, num_heads, dff)
            for _ in range(Nlayers)
        ]
        
        self.output_dense = tf.keras.layers.Dense(128)

    def call(self, inputs):
        mask, time = inputs
        
        maskp = self.input_proj(mask)
       
        positions = tf.cast(tf.range(sequence_length)[tf.newaxis, :],time.dtype) + time # shape (B, S)
        pos_enc = positional_encoding_from_indices(positions, d_model=self.d_model)  # returns (B, S, d_model)
        
        x = maskp[:,tf.newaxis,:] + pos_enc        
        for transformer_block in self.transformer_blocks:
            x = transformer_block(x)
        
        return self.output_dense(x)
    

@tf.function
def train_step(mask, time,target,optimizer):
    with tf.GradientTape() as tape:
        z = model([mask, time])                                    
        cosl = tf.reduce_mean(custom_cosine_similarity_loss(target, z))                    
        mse = tf.reduce_mean(tf.square(target - z)) 
        loss = cosl + mse 
    gradients = tape.gradient(loss, model.trainable_variables)
    optimizer.apply_gradients(zip(gradients, model.trainable_variables))
    return mse, cosl,loss

@tf.function
def val_step(mask, time,target,):
    z = model([mask, time])                                    
    cosl = tf.reduce_mean(custom_cosine_similarity_loss(target, z))                    
    mse = tf.reduce_mean(tf.square(target - z)) 
    loss = cosl + mse 
    return mse, cosl,loss

###############################################################################################################
####################################### Training Loop
def train_cvae(cvae, train_dataset, val_dataset,epochs=2000, patience=20, factor=0.25, min_lr=1e-9, saveM="cvae_model.keras",saveW="cvae.weights.h5", stop=50):
    #optimizer = tf.keras.optimizers.Adam(0.001)
    initial_lr=0.0001
    #initial_lr=0.00262
    optimizer = tf.keras.optimizers.Adam(learning_rate=initial_lr)
    nlr = initial_lr
    clip_value = 50.0 
    best_val_loss = float('inf')  # Track best validation loss
    num_Tbatches = len(data_loader.train_indices) // data_loader.batch_size  # Record for future epochs
    num_Vbatches = len(data_loader.val_indices) // data_loader.batch_size  # Record for future epochs
    early_stop_counter = 0
    for epoch in range(epochs):
        
        ###############################################################################################################
        ####################################### Training Phase
        train_loss = tmse = tkll = mur = logr = tcosl= 0
        print(f"\nEpoch {epoch + 1}/{epochs}: Training")
        with tqdm(total=num_Tbatches, desc="Progress", unit="batch") as pbar:
            for (mask,time),target in train_dataset:
                # mask, time_step, velocity = batch
                mse,cosl,loss = train_step(mask, time,target,optimizer)                                                             
                
                tcosl += cosl.numpy()
                train_loss += loss.numpy()
                tmse += mse.numpy()
                pbar.update(1)  # Update progress bar
        ###############################################################################################################
        ####################################### Validation Phase
        val_loss = vmse = vkll = vcosl= 0
        for (mask,time),target in val_dataset:        
            mse,cosl,loss = val_step(mask, time,target)    
            
            vcosl += cosl.numpy()
            val_loss += loss.numpy()
            vmse += mse.numpy()

        ###############################################################################################################
        ####################################### Compute Average Losses        
        train_loss /= num_Tbatches
        tmse /= num_Tbatches
        tcosl /= num_Tbatches
        val_loss /= num_Vbatches        
        vmse /= num_Vbatches
        vcosl /= num_Vbatches
        
        val_loss = vmse+vcosl
        
        # val_imp = best_val_loss-val_loss
        print(f"Epoch {epoch + 1}, Tcl: {tcosl:.3E}, Tmae: {tmse:.3E}, T Loss: {train_loss:.3E},  Best V Loss: {best_val_loss:.3E}") 
        print(f"Epoch {epoch + 1}, Vcl: {vcosl:.3E}, Vmae: {vmse:.3E}, V Loss: {val_loss:.3E}, NI: {early_stop_counter:.2f}, LR: {nlr:.2E}") 
                    
        nlr = reduce_lr_on_plateau(optimizer, val_loss, best_val_loss, patience, factor, min_lr)             

        ###############################################################################################################
        ####################################### Early Stopping Logic
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            early_stop_counter = 0  # Reset counter
            save_model_async(cvae, saveM,saveW)  # Save best model asynchronously
        else:
            early_stop_counter += 1
            if early_stop_counter >= stop:
                print(f"Early stopping at epoch {epoch + 1} due to no improvement.")
                break  # Stop training        

if __name__ == '__main__':
    # Initialize model
    model = TransformerVAE(
        d_model=d_model,
        num_heads=num_heads,
        Nlayers= Nlayers,
    )
    dummy_static = tf.random.normal((BS, mask_shape))
    dummy_dynamic = tf.random.normal((BS, 1))
    # 4. Build the model with explicit input shapes
    _= model([dummy_static, dummy_dynamic])

    # Now summary will work
    model.summary()
    '''
    try:
        model.load_weights('TE.weights.h5')
        #gan.discriminator.load_weights('discriminator.best.weights.h5')
        print('*********weight loaded')
    except ValueError as e:
        print(f"Error: {e}")
        raise  # Stop execution by re-raising the exception
    #'''
    #model.load_weights('TE.weights.h5')
    train_dataset = data_loader.get_dataset(mode="train")
    val_dataset = data_loader.get_dataset(mode="val")
    train_cvae(model, train_dataset, val_dataset,epochs=2000000, patience=51, factor=0.4, min_lr=1e-9, saveM="TE_model.keras",saveW="TE2.weights.h5", stop=500)

