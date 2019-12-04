import tensorflow as tf 

# define dataset configs and params
data_configs = tf.contrib.training.HParams(
    # general params
    speaker = "ngoc_anh_vov", # speaker name also folder contain data data/<speaker>
    sample_rate = 16000, # sample rate of input audio
    word_split = 'ConcatedVowels', # split word by charaters, vowels and consonants or somthing else
    # mfccs params
    winlen = 0.025, #the length of the analysis window in seconds. Default is 0.025s (25 milliseconds)
    winstep = 0.01, #the step between successive windows in seconds. Default is 0.01s (10 milliseconds)
    numcep = 13, #the number of cepstrum to return, default 13
    nfilt = 26, #the number of filters in the filterbank, default 26.
    nfft = 512, #the FFT size. Default is 512.
    lowfreq = 0, #lowest band edge of mel filters. In Hz, default is 0.
    highfreq = None, #highest band edge of mel filters. In Hz, default is samplerate/2
    preemph = 0.97, #apply preemphasis filter with preemph as coefficient. 0 is no filter. Default is 0.97.
    ceplifter = 22, #apply a lifter to final cepstral coefficients. 0 is no lifter. Default is 22.
    appendEnergy = False, #if this is true, the zeroth cepstral coefficient is replaced with the log of the total frame energy.
    # pitch features
    use_pitch = False,
    left_frames = 13,
    right_frames = 9
)

model_configs = tf.contrib.training.HParams(
    model_name = 'TDNN',
    subsample = True,
    # input params
    input_dim = 40, # base on extracted features, 40 for MFCCs only
    num_frames = 23, # number of frames need to compute 1 label
    output_dim = 183, # list characters size
    # layers params
    num_layers = 4, # number layers of model
    layer_info = [
        tf.contrib.training.HParams(
            layer_name = 'TDNN',
            context = [-2, -1, 0, 1, 2],
            num_filters = 512
        ),
        tf.contrib.training.HParams(
            layer_name = 'TDNN',
            context = [-1, 2],
            num_filters = 512
        ),
        tf.contrib.training.HParams(
            layer_name = 'TDNN',
            context = [-3, 3],
            num_filters = 512
        ),
        tf.contrib.training.HParams(
            layer_name = 'TDNN',
            context = [-7, 2],
            num_filters = 512
        )
    ],
    # general params
)

training_configs = tf.contrib.training.HParams(
    batch_size = 512,
    training_step = 200000,
    learning_rate = 1e-3,
    #Learning rate schedule
    decay_learning_rate = True, #boolean, determines if the learning rate will follow an exponential decay
    start_decay = 40000, #Step at which learning decay starts
    decay_steps = 18000, #Determines the learning rate decay slope (UNDER TEST)
    decay_rate = 0.5, #learning rate decay rate (UNDER TEST)
    initial_learning_rate = 1e-3, #starting learning rate
    final_learning_rate = 1e-4, #minimal learning rate
    # Optimization parameters
    adam_beta1 = 0.9, #AdamOptimizer beta1 parameter
    adam_beta2 = 0.999, #AdamOptimizer beta2 parameter
    adam_epsilon = 1e-6, #AdamOptimizer Epsilon parameter
    # log params
    train_epochs = 25
)
