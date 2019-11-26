import tensorflow as tf 

# define dataset configs and params
data_configs = tf.contrib.training.HParams(
    # general configs
    speaker = "ngoc_anh_vov", # speaker name also folder contain data data/<speaker>
    sample_rate = 16000, # sample rate of input audio
    word_split = 'ConcatedVowels', # split word by charaters, vowels and consonants or somthing else
    # mfccs configs
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
    use_pitch = False
)

model_configs = {}

training_configs = {}
