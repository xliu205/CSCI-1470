import tensorflow as tf
from tensorflow.keras import Sequential
from tensorflow.keras.layers import Dense, Flatten, Reshape
from tensorflow.math import exp, sqrt, square


class VAE(tf.keras.Model):
    def __init__(self, input_size, latent_size=15):
        super(VAE, self).__init__()
        self.input_size = input_size # H*W
        self.latent_size = latent_size  # Z

        self.hidden_dim = 400

        self.encoder = tf.keras.Sequential(
            [
                tf.keras.layers.Dense(self.hidden_dim, input_shape=(128, self.input_size), activation="relu"),
                tf.keras.layers.Dense(self.hidden_dim, activation="relu"),
                tf.keras.layers.Dense(self.hidden_dim, activation="relu"),
            ]
        )

        self.mu_layer = tf.keras.layers.Dense(self.latent_size)
        self.logvar_layer = tf.keras.layers.Dense(self.latent_size)

        self.decoder = tf.keras.Sequential(
            [
                tf.keras.layers.Dense(self.hidden_dim, activation="relu"),
                tf.keras.layers.Dense(self.hidden_dim, activation="relu"),
                tf.keras.layers.Dense(self.hidden_dim, activation="relu"),
                tf.keras.layers.Dense(self.input_size, activation="sigmoid"),
            ]
        )

        self.optimizer = tf.keras.optimizers.Adam(learning_rate=1e-3)

    def call(self, x):
        """
        Performs forward pass through FC-VAE model by passing image through 
        encoder, reparametrize trick, and decoder models
    
        Inputs:
        - x: Batch of input images of shape (N, 1, H, W)
        
        Returns:
        - x_hat: Reconstruced input data of shape (N,1,H,W)
        - mu: Matrix representing estimated posterior mu (N, Z), with Z latent space dimension
        - logvar: Matrix representing estimataed variance in log-space (N, Z), with Z latent space dimension
        """
        encoder = self.encoder(x)
        mu = self.mu_layer(encoder)
        logvar = self.logvar_layer(encoder)
        z = reparametrize(mu, logvar)
        decoder = self.decoder(z)
        x_hat = decoder

        return x_hat, mu, log_var


class CVAE(tf.keras.Model):
    def __init__(self, input_size, num_classes=10, latent_size=15):
        super(CVAE, self).__init__()
        self.input_size = input_size # H*W
        self.latent_size = latent_size # Z
        self.num_classes = num_classes # C

        self.hidden_dim = 400
        self.encoder = tf.keras.Sequential(
            [
                tf.keras.layers.Dense(self.hidden_dim, input_shape=(128, self.input_size + self.num_classes), activation="relu"),
                tf.keras.layers.Dense(self.hidden_dim, activation="relu"),
                tf.keras.layers.Dense(self.hidden_dim, activation="relu"),
            ]
        )

        self.mu_layer = tf.keras.layers.Dense(self.latent_size)
        self.logvar_layer = tf.keras.layers.Dense(self.latent_size)

        self.decoder = tf.keras.Sequential(
            [
                tf.keras.layers.Dense(self.hidden_dim, activation="relu"),
                tf.keras.layers.Dense(self.hidden_dim, activation="relu"),
                tf.keras.layers.Dense(self.hidden_dim, activation="relu"),
                tf.keras.layers.Dense(self.input_size, activation="sigmoid"),
            ]
        )

        self.optimizer = tf.keras.optimizers.Adam(learning_rate=1e-3)


    def call(self, x, c):
        """
        Performs forward pass through FC-CVAE model by passing image through 
        encoder, reparametrize trick, and decoder models
    
        Inputs:
        - x: Input data for this timestep of shape (N, 1, H, W)
        - c: One hot vector representing the input class (0-9) (N, C)
        
        Returns:
        - x_hat: Reconstruced input data of shape (N, 1, H, W)
        - mu: Matrix representing estimated posterior mu (N, Z), with Z latent space dimension
        - logvar: Matrix representing estimated variance in log-space (N, Z),  with Z latent space dimension
        """
        x_shape = x.shape
        x = tf.keras.layers.Flatten()(x)
        x = tf.concat([x,c],1)
        encoder = self.encoder(x)
        mu = self.mu_layer(encoder)
        logvar = self.logvar_layer(encoder)
        z = reparametrize(mu, logvar)
        z = tf.concat9([z,c],1)
        decoder = self.decoder(z)
        x_hat = tf.reshape(self.decoder(z),x_shape)
        return x_hat, mu, log_var


def reparametrize(mu, logvar):
    """
    Differentiably sample random Gaussian data with specified mean and variance using the
    reparameterization trick.

    Suppose we want to sample a random number z from a Gaussian distribution with mean mu and
    standard deviation sigma, such that we can backpropagate from the z back to mu and sigma.
    We can achieve this by first sampling a random value epsilon from a standard Gaussian
    distribution with zero mean and unit variance, then setting z = sigma * epsilon + mu.

    For more stable training when integrating this function into a neural network, it helps to
    pass this function the log of the variance of the distribution from which to sample, rather
    than specifying the standard deviation directly.

    Inputs:
    - mu: Tensor of shape (N, Z) giving means
    - logvar: Tensor of shape (N, Z) giving log-variances

    Returns: 
    - z: Estimated latent vectors, where z[i, j] is a random value sampled from a Gaussian with
         mean mu[i, j] and log-variance logvar[i, j].
    """
    N, Z = mu.shape
    var = exp(logvar)
    epsilon = tf.random.normal((N,Z),dtype= mu.shape)
    z = sqrt(var)* epsilon + mu

    ################################################################################################
    #                              END OF YOUR CODE                                                #
    ################################################################################################
    return z


def bce_function(x_hat, x):
    """
    Computes the reconstruction loss of the VAE.
    
    Inputs:
    - x_hat: Reconstructed input data of shape (N, 1, H, W)
    - x: Input data for this timestep of shape (N, 1, H, W)
    
    Returns:
    - reconstruction_loss: Tensor containing the scalar loss for the reconstruction loss term.
    """
    bce_fn = tf.keras.losses.BinaryCrossentropy(
        from_logits=False, 
        reduction=tf.keras.losses.Reduction.SUM,
    )
    reconstruction_loss = bce_fn(x, x_hat) * x.shape[-1]  # Sum over all loss terms for each data point. This looks weird, but we need this to work...
    return reconstruction_loss


def loss_function(x_hat, x, mu, logvar):
    """
    Computes the negative variational lower bound loss term of the VAE (refer to formulation in notebook).
    Returned loss is the average loss per sample in the current batch.

    Inputs:
    - x_hat: Reconstructed input data of shape (N, 1, H, W)
    - x: Input data for this timestep of shape (N, 1, H, W)
    - mu: Matrix representing estimated posterior mu (N, Z), with Z latent space dimension
    - logvar: Matrix representing estimated variance in log-space (N, Z), with Z latent space dimension
    
    Returns:
    - loss: Tensor containing the scalar loss for the negative variational lowerbound
    """
    reconstruction_loss = bce_function(x_hat, x)
    kl_loss = -0.5 * tf.reduce_sum(1 + logvar - mu ** 2 - tf.math.exp(logvar))
    return （reconstruction_loss + kl_loss） / float(x_hat.shape[0])
