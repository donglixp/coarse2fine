def model_opts(parser):
    """
    These options are passed to the construction of the model.
    Be careful with these as they will be used during translation.
    """
    # Model options
    # Embedding Options
    parser.add_argument('-word_vec_size', type=int, default=300,
                        help='Word embedding for both.')
    parser.add_argument('-ent_vec_size', type=int, default=10,
                        help='POS embedding size.')

    # RNN Options
    parser.add_argument('-encoder_type', type=str, default='brnn',
                        choices=['rnn', 'brnn'],
                        help="""Type of encoder layer to use.""")
    parser.add_argument('-decoder_type', type=str, default='rnn',
                        choices=['rnn', 'transformer', 'cnn'],
                        help='Type of decoder layer to use.')

    parser.add_argument('-layers', type=int, default=1,
                        help='Number of layers in enc/dec.')
    parser.add_argument('-enc_layers', type=int, default=1,
                        help='Number of layers in the encoder')
    parser.add_argument('-dec_layers', type=int, default=1,
                        help='Number of layers in the decoder')

    parser.add_argument('-rnn_size', type=int, default=250,
                        help='Size of LSTM hidden states')
    parser.add_argument('-score_size', type=int, default=64,
                        help='Size of hidden layer in scorer')

    parser.add_argument('-rnn_type', type=str, default='LSTM',
                        choices=['LSTM', 'GRU'],
                        help="""The gate type to use in the RNNs""")
    parser.add_argument('-brnn_merge', default='concat',
                        choices=['concat', 'sum'],
                        help="Merge action for the bidir hidden states")

    # Table encoding options
    parser.add_argument('-split_type', default='incell',
                        choices=['incell', 'outcell'],
                        help="whether encode column split token |")
    parser.add_argument('-merge_type', default='cat',
                        choices=['sub', 'cat', 'mlp'],
                        help="compute span vector for table column: mlp>cat>sub")

    # Decoder options
    parser.add_argument('-layout_encode', default='rnn',
                        choices=['none', 'rnn'],
                        help="Layout encoding method.")
    parser.add_argument('-cond_op_vec_size', type=int, default=150,
                        help='Layout embedding size.')

    # Attention options
    parser.add_argument('-global_attention', type=str, default='general',
                        choices=['dot', 'general', 'mlp'],
                        help="""The attention type to use:
                        dotprot or general (Luong) or MLP (Bahdanau)""")
    parser.add_argument('-attn_hidden', type=int, default=64,
                        help="if attn_hidden > 0, then attention score = f(Ue) B f(Ud)")
    parser.add_argument('-co_attention', action="store_true",
                        help="if attn_hidden > 0, then attention score = f(Ue) B f(Ud)")


def preprocess_opts(parser):
    # Dictionary Options
    parser.add_argument('-src_vocab_size', type=int, default=100000,
                        help="Size of the source vocabulary")
    parser.add_argument('-src_words_min_frequency', type=int, default=0)

    # Truncation options
    parser.add_argument('-src_seq_length', type=int, default=50,
                        help="Maximum source sequence length")
    parser.add_argument('-src_seq_length_trunc', type=int, default=0,
                        help="Truncate source sequence length.")
    parser.add_argument('-tgt_seq_length', type=int, default=50,
                        help="Maximum target sequence length to keep.")
    parser.add_argument('-tgt_seq_length_trunc', type=int, default=0,
                        help="Truncate target sequence length.")

    # Data processing options
    parser.add_argument('-shuffle', type=int, default=1,
                        help="Shuffle data")
    parser.add_argument('-lower', action='store_true', help='lowercase data')

    parser.add_argument('-span_exact_match', action="store_true",
                        help='Must have exact match for cond span in WHERE clause')


def train_opts(parser):
    # Model loading/saving options
    parser.add_argument('-data', default='',
                        help="""Path prefix to the "train.pt" and
                        "valid.pt" file path from preprocess.py""")
    parser.add_argument('-save_dir', default='',
                        help="Model save dir")
    parser.add_argument('-train_from', default='', type=str,
                        help="""If training from a checkpoint then this is the
                        path to the pretrained model's state_dict.""")
    # GPU
    parser.add_argument('-gpuid', default=[0], nargs='+', type=int,
                        help="Use CUDA on the listed devices.")
    parser.add_argument('-seed', type=int, default=123,
                        help="""Random seed used for the experiments
                        reproducibility.""")

    # Init options
    parser.add_argument('-start_epoch', type=int, default=1,
                        help='The epoch from which to start')
    parser.add_argument('-param_init', type=float, default=0.08,
                        help="""Parameters are initialized over uniform distribution
                        with support (-param_init, param_init).
                        Use 0 to not use initialization""")

    parser.add_argument('-fix_word_vecs', action='store_true',
                        help="Fix word embeddings on the encoder side.")
    parser.add_argument('-update_word_vecs_after', type=int, default=10,
                        help='When fix_word_vecs=True, only update word vectors after update_word_vecs_after epochs.')
    parser.add_argument('-agg_sample_rate', type=float, default=0.5,
                        help='Randomly skip agg loss, because this loss term tends to be overfitting.')

    # Optimization options
    parser.add_argument('-batch_size', type=int, default=200,
                        help='Maximum batch size')
    parser.add_argument('-max_generator_batches', type=int, default=32,
                        help="""Maximum batches of words in a sequence to run
                        the generator on in parallel. Higher is faster, but
                        uses more memory.""")
    parser.add_argument('-epochs', type=int, default=40,
                        help='Number of training epochs')
    parser.add_argument('-optim', default='rmsprop',
                        choices=['sgd', 'adagrad',
                                 'adadelta', 'adam', 'rmsprop'],
                        help="""Optimization method.""")
    parser.add_argument('-max_grad_norm', type=float, default=5,
                        help="""If the norm of the gradient vector exceeds this,
                        renormalize it to have the norm equal to
                        max_grad_norm""")
    parser.add_argument('-dropout', type=float, default=0.5,
                        help="Dropout rate.")
    parser.add_argument('-lock_dropout', action='store_true',
                        help="Use the same dropout mask for RNNs.")
    parser.add_argument('-weight_dropout', type=float, default=0,
                        help=">0: Weight dropout probability; applied in LSTM stacks.")
    parser.add_argument('-smooth_eps', type=float, default=0,
                        help="Label smoothing")
    # learning rate
    parser.add_argument('-learning_rate', type=float, default=0.002,
                        help="""Starting learning rate.""")
    parser.add_argument('-alpha', type=float, default=0.95,
                        help="Optimization hyperparameter")
    parser.add_argument('-learning_rate_decay', type=float, default=0.98,
                        help="""If update_learning_rate, decay learning rate by this much if (i) perplexity does not decrease on the validation set or (ii) epoch has gone past start_decay_at""")
    parser.add_argument('-start_decay_at', type=int, default=8,
                        help="""Start decaying every epoch after and including this epoch""")
    parser.add_argument('-start_checkpoint_at', type=int, default=30,
                        help="""Start checkpointing every epoch after and including this epoch""")
    parser.add_argument('-decay_method', type=str, default="",
                        choices=['noam'], help="Use a custom decay rate.")
    parser.add_argument('-warmup_steps', type=int, default=4000,
                        help="""Number of warmup steps for custom decay.""")

    parser.add_argument('-report_every', type=int, default=50,
                        help="Print stats at this interval.")
    parser.add_argument('-exp', type=str, default="",
                        help="Name of the experiment for logging.")


def translate_opts(parser):
    parser.add_argument('-model_path', required=True,
                        help='Path to model .pt file')
    parser.add_argument('-data_path', default='',
                        help='Path to data')
    parser.add_argument('-split', default="dev",
                        help="Path to the evaluation annotated data")
    parser.add_argument('-output', default='pred.txt',
                        help="""Path to output the predictions (each line will be the decoded sequence""")
    parser.add_argument('-batch_size', type=int, default=30,
                        help='Batch size')
    parser.add_argument('-gpu', type=int, default=0,
                        help="Device to run on")
    parser.add_argument('-gold_layout', action='store_true',
                        help="Given the golden layout sequences for evaluation.")
