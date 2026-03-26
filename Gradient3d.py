import tensorflow as tf
def gradient3d(x, spacing=(1., 1., 1.)):
    """
    NumPy-exact ∇ for 4-D [H,W,D,C] or 5-D [N,H,W,D,C] tensors.
    spacing = (dy, dx, dz) voxel size.
    Returns dy, dx, dz – each same shape as x.
    """
    x = tf.convert_to_tensor(x)
    add_batch = (x.shape.rank == 4)          # add fake batch if needed
    if add_batch:
        x = tf.expand_dims(x, 0)             # -> [1,H,W,D,C]

    dx = tf.concat([
        (x[:,1:2,:,:,:]   - x[:,0:1,:,:,:]) / spacing[0],      # forward
        (x[:,2:,:,:,:]    - x[:,:-2,:,:,:]) / (2*spacing[0]),  # central
        (x[:,-1:,:,:,:]   - x[:,-2:-1,:,:,:]) / spacing[0]     # backward
    ], axis=1)

    dy = tf.concat([
        (x[:,:,1:2,:,:]   - x[:,:,0:1,:,:]) / spacing[1],
        (x[:,:,2:,:,:]    - x[:,:,:-2,:,:]) / (2*spacing[1]),
        (x[:,:,-1:,:,:]   - x[:,:,-2:-1,:,:]) / spacing[1]
    ], axis=2)

    dz = tf.concat([
        (x[:,:,:,1:2,:]   - x[:,:,:,0:1,:]) / spacing[2],
        (x[:,:,:,2:,:]    - x[:,:,:,:-2,:]) / (2*spacing[2]),
        (x[:,:,:,-1:,:]   - x[:,:,:,-2:-1,:]) / spacing[2]
    ], axis=3)

    if add_batch:
        dx, dy, dz = [t[0] for t in (dx, dy, dz)]  # strip fake batch
    return tf.abs(dx), tf.abs(dy), tf.abs(dz)
