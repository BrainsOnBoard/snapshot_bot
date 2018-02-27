import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf

from cv2 import imread, resize, IMREAD_GRAYSCALE
from glob import glob

#-----------------------------------------------------------------------------
# Parameters
#-----------------------------------------------------------------------------
SNAPSHOT_WIDTH = 90#69
SNAPSHOT_HEIGHT = 25

# Calculate layer sizes
#LAYER_SIZES = [SNAPSHOT_WIDTH * SNAPSHOT_HEIGHT, 3072, 1024, 256, 64]
LAYER_SIZES = [SNAPSHOT_WIDTH * SNAPSHOT_HEIGHT, 1024, 512, 128, 64]

# Training Parameters
LEARNING_RATE = 0.001
BATCH_SIZE = 100
DISPLAY_STEP = 250

NUM_STEPS = 10000

# How many training images should we keep behind for testing
# **NOTE** this is kinda unnecessary as the
NUM_TESTING = 10

# How many in-silica 'rolls' should we make of each training image - this generates rotated versions
NUM_IMAGE_ROLLS = 1
IMAGE_ROLL_PIXELS = SNAPSHOT_WIDTH / NUM_IMAGE_ROLLS

# Should we train the network or just load the exported model
TRAIN = True

#-----------------------------------------------------------------------------
# Entry point
#-----------------------------------------------------------------------------
# Search for route data
training_images = glob("office_training_3/*.png")

# Create empty array of training data
data = np.empty(dtype=np.float32, shape=(len(training_images) * NUM_IMAGE_ROLLS, LAYER_SIZES[0]))

# Load route images into rows
for i, t in enumerate(training_images):
    # Read image
    image = imread(t, IMREAD_GRAYSCALE).astype(np.float32)
    #image = image[:,15:84]
    image /= 255.0

    # Copy into first row corresponding to this image
    data[i * NUM_IMAGE_ROLLS] = np.reshape(image, LAYER_SIZES[0])

    # Loop through in-silica 'rolls'
    for r in range(1, NUM_IMAGE_ROLLS):
        # Roll image suitable amount
        rolled_image = np.roll(image, r * IMAGE_ROLL_PIXELS, axis=1)
        data[(i * NUM_IMAGE_ROLLS) + r] = np.reshape(rolled_image, LAYER_SIZES[0])

# Shuffle dataset
# **NOTE** this only shuffles along 1st axis as required
np.random.shuffle(data)

# Split into testing and training
testing_data = data[:NUM_TESTING]
training_data = data[NUM_TESTING:]

# Start Training
# Start a new TF session
with tf.Session() as sess:
    if TRAIN:
        # tf Graph input (only pictures)
        X = tf.placeholder("float", [None, LAYER_SIZES[0]], name="input")

        # **NOTE** 'Xavier' initialization speeds up learning immensely
        initializer = tf.contrib.layers.xavier_initializer()
        encoder_weights = [tf.Variable(initializer([layer_size, next_layer_size]))
                           for layer_size, next_layer_size in zip(LAYER_SIZES[:-1], LAYER_SIZES[1:])]
        decoder_weights = [tf.Variable(initializer([layer_size, next_layer_size]))
                           for layer_size, next_layer_size in reversed(zip(LAYER_SIZES[1:], LAYER_SIZES[:-1]))]

        encoder_biases = [tf.Variable(initializer([layer_size]))
                          for layer_size in LAYER_SIZES[1:]]
        decoder_biases = [tf.Variable(initializer([layer_size]))
                          for layer_size in reversed(LAYER_SIZES[:-1])]

        # Building the encoder
        def encoder(x):
            layers = []
            for i, (w, b) in enumerate(zip(encoder_weights, encoder_biases)):
                inp = x if i == 0 else layers[i - 1]
                layers.append(tf.nn.sigmoid(tf.add(tf.matmul(inp, w), b), name="encoder_%u" % i))
            return layers[-1]


        # Building the decoder
        def decoder(x):
            layers = []
            for i, (w, b) in enumerate(zip(decoder_weights, decoder_biases)):
                inp = x if i == 0 else layers[i - 1]
                layers.append(tf.nn.sigmoid(tf.add(tf.matmul(inp, w), b), name="decoder_%u" % i))
            return layers[-1]


        # Construct model
        encoder_op = encoder(X)
        decoder_op = decoder(encoder_op)

        # Prediction
        y_pred = decoder_op

        # Targets (Labels) are the input data.
        y_true = X

        # Define loss and optimizer, minimize the squared error
        loss = tf.reduce_mean(tf.pow(y_true - y_pred, 2))
        optimizer = tf.train.AdamOptimizer(learning_rate=LEARNING_RATE).minimize(loss)

        # Initialize the variables (i.e. assign their default value)
        init = tf.global_variables_initializer()

        # Run the initializer
        sess.run(init)

        print("Training (ctrl-c to stop)")
        try:
            # Training
            index_in_epoch = 0
            for i in range(NUM_STEPS):
                start = index_in_epoch

                # Go to the next epoch
                batch_x = None
                if (index_in_epoch + BATCH_SIZE) > training_data.shape[0]:
                    # Get the rest examples in this epoch
                    rest_num_examples = training_data.shape[0] - index_in_epoch
                    images_rest_part = training_data[start:training_data.shape[0]]

                    # Shuffle dataset
                    # **NOTE** this only shuffles along 1st axis as required
                    np.random.shuffle(training_data)

                    # Start next epoch
                    start = 0
                    index_in_epoch = BATCH_SIZE - rest_num_examples
                    end = index_in_epoch
                    images_new_part = training_data[start:end]

                    batch_x = np.concatenate((images_rest_part, images_new_part), axis=0)
                else:
                    index_in_epoch += BATCH_SIZE
                    end = index_in_epoch
                    batch_x = training_data[start:end]

                # Run optimization op (backprop) and cost op (to get loss value)
                _, l = sess.run([optimizer, loss], feed_dict={X: batch_x})

                if i == 0 or i % DISPLAY_STEP == 0:
                    print("Step %u, Minibatch loss:%f" % (i, l))
        except KeyboardInterrupt:
            pass

        # Saving
        builder = tf.saved_model.builder.SavedModelBuilder("./export")
        builder.add_meta_graph_and_variables(sess, ["tag"],
                                             signature_def_map=None,
                                             assets_collection=None)
        builder.save()
        print("Saved model")
    else:
        tf.saved_model.loader.load(sess, ["tag"], "./export")
        graph = tf.get_default_graph()
        X = graph.get_tensor_by_name("input:0")
        decoder_op = graph.get_tensor_by_name("decoder_2:0")
        print("Model restored")

    print("Testing")
    reconstructed = sess.run(decoder_op, feed_dict={X: testing_data})

    # Plot results
    fig, axes = plt.subplots(NUM_TESTING, 2)
    for i in range(NUM_TESTING):
        input_image = np.reshape(testing_data[i], (SNAPSHOT_HEIGHT, SNAPSHOT_WIDTH))
        reconstructed_image = np.reshape(reconstructed[i], (SNAPSHOT_HEIGHT, SNAPSHOT_WIDTH))

        axes[i, 0].imshow(input_image, cmap="gray")
        axes[i, 1].imshow(reconstructed_image, cmap="gray")
    plt.show()