import numpy as np
import os
import tensorflow as tf
import keras
import random
from tensorflow.keras import layers, losses, models
from keras.models import Model, load_model
from keras.callbacks import ModelCheckpoint, EarlyStopping, ReduceLROnPlateau, CSVLogger, Callback
from sklearn.model_selection import train_test_split
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from PrecisionRecallF1ScoreMetrics import *
from BinaryFocalLossNDice2 import * #combined_weighted_focal_dice_loss
import threading
from tqdm import tqdm
# ==========================================================================================================
os.environ["TF_GPU_ALLOCATOR"] = "cuda_malloc_async"
# Enable TF XLA JIt compilation - dynamically chose best-performing algorithm
os.environ['TF_XLA_FLAGS'] = '--tf_xla_auto_jit=2'
# =========================================================================================================
tf.config.optimizer.set_jit(True)
#tf.config.set_visible_devices([], 'GPU')                      # Disable GPU
#tf.keras.mixed_precision.set_global_policy('mixed_float16')   # Use float16
############################################################################################################            
# Define input dimensions   ################################################################################
Name = 'FM1B'
BS = 4
#filters = [32,16,8,4]
filters = [4,8,16,32,64]
enc_in_shape = (176,80,64,1)
dec_in_shape = (11,5,4,filters[-1]) 
kernel_size=3
act = 'relu'
save_every_n_epochs = 100
latent_dim = 128
############################################################################################################
# Asynchronous Model Saving Function
def save_model_async(model, saveM,saveW):
    def save():
        model.save(saveM)
        model.save_weights(saveW)
        print(f"Model saved at {saveM and saveW}.")
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
    """Custom function to clip logvar within [-5, 5] inside a Keras layer."""
    return tf.clip_by_value(x, -10, 3)
############################################################################################################  
# Build the encoder

def cvae_encoder(x,latent_dim):
    # Encoder
    c1 = layers.Conv3D(filters[0], 3, activation='relu', padding='same')(x)
    c1 = layers.Conv3D(filters[0], 3, activation='relu', padding='same')(c1)
    p1 = layers.MaxPool3D(2)(c1)

    c2 = layers.Conv3D(filters[1], 3, activation='relu', padding='same')(p1)
    c2 = layers.Conv3D(filters[1], 3, activation='relu', padding='same')(c2)
    p2 = layers.MaxPool3D(2)(c2)

    c3 = layers.Conv3D(filters[2], 3, activation='relu', padding='same')(p2)
    c3 = layers.Conv3D(filters[2], 3, activation='relu', padding='same')(c3)
    p3 = layers.MaxPool3D(2)(c3)
    
    c4 = layers.Conv3D(filters[3], 3, activation='relu', padding='same')(p3)
    c4 = layers.Conv3D(filters[3], 3, activation='relu', padding='same')(c4)
    p4 = layers.MaxPool3D(2)(c4)

    bn = layers.Conv3D(filters[4], 3, activation='relu', padding='same')(p4)
    bn = layers.Flatten()(bn)

    x = layers.Dense(512)(bn)
    x = layers.ReLU()(x)

    mu = layers.Dense(latent_dim, name='mu')(x)
    logvar = layers.Dense(latent_dim)(x)
#    logvar = layers.Activation(clip_logvar)(logvar)
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
    x = layers.Dense(11*5*4*filters[4],activation='relu')(z)
    x = layers.Reshape((11,5,4,filters[4]))(x)
    x = layers.Conv3D(filters[4], 3, activation='relu', padding='same')(x)

    x = layers.Conv3DTranspose(filters[3], 3, strides=2, padding='same')(x)
    x = layers.Conv3D(filters[3], 3, activation='relu', padding='same')(x)
    
    x = layers.Conv3DTranspose(filters[2], 3, strides=2, padding='same')(x)
    x = layers.Conv3D(filters[2], 3, activation='relu', padding='same')(x)

    x = layers.Conv3DTranspose(filters[1], 3, strides=2, padding='same')(x)
    x = layers.Conv3D(filters[1], 3, activation='relu', padding='same')(x)

    x = layers.Conv3DTranspose(filters[0], 3, strides=2, padding='same')(x)
    x = layers.Conv3D(filters[0], 3, activation='relu', padding='same')(x)

    return layers.Conv3D(1, 1, activation='sigmoid')(x)

def cvae_model(input_shape,  latent_dim):
    """CVAE Model"""
    x_input = tf.keras.Input(shape=input_shape)

    mu, logvar = cvae_encoder(x_input, latent_dim)
    z = sample_latent(mu, logvar)
    decoded = cvae_decoder(z)
    return tf.keras.Model(inputs=x_input, outputs=[decoded, mu, logvar])

# Loss Function
def cvae_loss(x, x_recon, mu, logvar,beta=1):
    recon_loss = combined_weighted_focal_dice_loss(x, x_recon,alpha=0.98, gamma=2.0, foreground_weight=5.0, focal_weight=1.0, dice_weight=1.0)
    #recon_loss = tf.reduce_mean(tf.keras.losses.MAE(x, x_recon))  # Reconstruction loss
    kl_loss = -0.5 * tf.reduce_mean(1 + logvar - tf.square(mu) - tf.exp(logvar))  # KL divergence
    return (100000*recon_loss) + (beta*kl_loss), recon_loss, kl_loss

###############################################################################################################
####################################### Training Loop
def train_cvae(cvae, train_dataset, val_dataset,epochs=2000, patience=20, factor=0.25, min_lr=1e-9, saveM="cvae_model.keras",saveW="cvae.weights.h5", stop=50):
    #optimizer = tf.keras.optimizers.Adam(0.001)
    initial_lr=0.0005
    #initial_lr=0.00262
    optimizer = tf.keras.optimizers.Adam(learning_rate=initial_lr)
    tf1 = tf.keras.metrics.F1Score(name="tf1")
    vf1 = tf.keras.metrics.F1Score(name="vf1")
    tacc = tf.keras.metrics.Accuracy(name="tacc")
    vacc = tf.keras.metrics.Accuracy(name="vacc")
    tp = tf.keras.metrics.Precision(name="tp")
    vp = tf.keras.metrics.Precision(name="vp")
    tr = tf.keras.metrics.Recall(name="tr")
    vr = tf.keras.metrics.Recall(name="vr")
    nlr = initial_lr
    clip_value = 50.0 
    best_val_loss = float('inf')  # Track best validation loss
    bch = 4
    num_Tbatches = x_train.shape[0]//bch
    num_Vbatches = x_test.shape[0]//bch
    
    for epoch in range(epochs):
        tf1.reset_state()
        vf1.reset_state()
        tacc.reset_state()
        vacc.reset_state()
        tp.reset_state()
        vp.reset_state()
        tr.reset_state()
        vr.reset_state()
        
        ###############################################################################################################
        ####################################### Training Phase
        train_loss = tmse = tkll = mur = logr = 0
        print(f"\nEpoch {epoch + 1}/{epochs}: Training")
        with tqdm(total=num_Tbatches, desc="Progress", unit="batch") as pbar:
            for xtrain in train_dataset:                
                # mask, time_step, velocity = batch
                with tf.GradientTape() as tape:
                    x_recon, mu, logvar = cvae(xtrain)
                    loss, mse, kll = cvae_loss(xtrain, x_recon, mu, logvar)
                    
                gradients = tape.gradient(loss, cvae.trainable_variables)
#                gradients, _ = tf.clip_by_global_norm(gradients, clip_value)  # Clip gradients
                optimizer.apply_gradients(zip(gradients, cvae.trainable_variables))
                
#                tf1.update_state(xtrain,x_recon)
                tacc.update_state(xtrain,x_recon)
                tp.update_state(xtrain,x_recon)
                tr.update_state(xtrain,x_recon)
                
                train_loss += loss.numpy()               
                tmse += mse.numpy()
                tkll += kll.numpy()
                mur += tf.reduce_mean(mu).numpy()
                logr += tf.reduce_mean(logvar).numpy()
                pbar.update(1)  # Update progress bar
        ###############################################################################################################
        ####################################### Validation Phase
        val_loss = vmse = vkll = 0
        for xtest in val_dataset:            
            x_recon_val, mu_val, logvar_val = cvae(xtest)
            loss, mse, kll = cvae_loss(xtest, x_recon_val, mu_val, logvar_val)
            val_loss += loss.numpy()
            vmse += mse.numpy()
            vkll += kll.numpy()

#            vf1.update_state(xtrain,x_recon)
            vacc.update_state(xtrain,x_recon)
            vp.update_state(xtrain,x_recon)
            vr.update_state(xtrain,x_recon)

        ###############################################################################################################
        ####################################### Compute Average Losses        
        train_loss /= num_Tbatches
        tmse /= num_Tbatches
        tkll /= num_Tbatches
        mur /= num_Tbatches
        logr /= num_Tbatches
        val_loss /= num_Vbatches        
        vmse /= num_Vbatches
        vkll /= num_Vbatches
        
        val_loss = vmse
        
        print(f"Tmae: {tmse:.4f}, Tkll: {tkll:.4f}, T Loss: {train_loss:.4f},Tacc: {tacc.result().numpy():.4f}, Tp: {tp.result().numpy():.4f}, Tr: {tr.result().numpy():.4f}") #, TF1: {tf1.result().numpy():.4f}") 
        print(f"Vmae: {vmse:.4f}, Vkll: {vkll:.4f}, V Loss: {val_loss:.4f}  ,Vacc: {vacc.result().numpy():.4f}, Vp: {vp.result().numpy():.4f}, Vr: {vr.result().numpy():.4f}") #, VF1: {vf1.result().numpy():.4f}") 
        
        val_imp = best_val_loss-val_loss
        
        print(f"mu: {mur:.4E}, logvar: {logr:.4E}, ValImp: {val_imp:.4f}, Best V Loss: {best_val_loss:.4f}, LR: {nlr:.2E}")
        
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
############################################################################################################
# Train   ##################################################################################################
data = np.load('/home/jseeyave3/FluidMasks/DAtrain1B.AngSweep.FM.npy')
x_train, x_test = train_test_split(data, test_size=0.2, random_state=69)
del data
x_train = tf.cast(x_train, tf.float16)
x_test  = tf.cast(x_test,  tf.float16)
train_dataset = tf.data.Dataset.from_tensor_slices((x_train)).batch(BS)
val_dataset = tf.data.Dataset.from_tensor_slices((x_test)).batch(BS)

model = cvae_model(enc_in_shape, latent_dim)
model.summary()

train_cvae(model, train_dataset, val_dataset,epochs=2000, patience=8, factor=0.5, min_lr=1e-9, saveM="cvae21_model.keras",saveW="cvae21.weights.h5", stop=60)
#model.save('vae.d'+str(latent_dim)+'.keras')
#model.save_weights('vae.d'+str(latent_dim)+'.weights.h5')

'''
data = np.load('/home/jseeyave3/FluidMasks/DAtrain1B.AngSweep.FM.npy')
model.load_weights('vae.d'+str(latent_dim)+'.weights.h5')
pred,_,_ = model.predict(data)
print('pred',pred.shape)
np.save('prediction.d'+str(latent_dim),pred) #'''

############################################################################################################
# Extract latent from trained model ########################################################################
model.load_weights('cvae21.weights.h5')
# latent model
bottleneck_model = tf.keras.Model(
    inputs=model.input,
    outputs=model.get_layer('mu').output
)
# Training latent to train transformer
data = np.load('/home/jseeyave3/FluidMasks/Train1BFM.AngSweep.npy')
latent_features = bottleneck_model.predict(data)
print("Latent features shape:", latent_features.shape)
np.save('trainFMLatent.d'+str(latent_dim),latent_features)
# test latent
data = np.load('/home/jseeyave3/FluidMasks/Test1BFM.AngSweep.npy')
latent_features = bottleneck_model.predict(data)
print("Latent features shape:", latent_features.shape)
np.save('testFMLatent.d'+str(latent_dim),latent_features)

