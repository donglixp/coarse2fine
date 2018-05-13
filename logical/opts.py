import argparse


def model_opts(parser):
    """
    These options are passed to the construction of the model.
    Be careful with these as they will be used during translation.
    """
    # Model options
    # Embedding Options
    parser.add_argument('-word_vec_size', type=int, default=150,
                        help='Word embedding for both.')
    parser.add_argument('-ent_vec_size', type=int, default=0,
                        help='Entity type embedding size.')
    parser.add_argument('-decoder_input_size', type=int, default=150,
                        help='Layout embedding size.')

    # RNN Options
    parser.add_argument('-seprate_encoder', action="store_true",
                        help="Use different encoders for layout and target decoding.")
    parser.add_argument('-encoder_type', type=str, default='brnn',
                        choices=['rnn', 'brnn'],
                        help="""Type of encoder layer to use.""")
    parser.add_argument('-decoder_type', type=str, default='rnn',
                        choices=['rnn'],
                        help='Type of decoder layer to use.')
    parser.add_argument('-parent_feed', type=str, default='output',
                        choices=['none', 'input', 'output'],
                        help="Feeding parent vector into the current time step of decoder.")
    parser.add_argument('-parent_feed_hidden', type=int, default=0,
                        help='Hidden size for parent feed vector.')

    parser.add_argument('-layers', type=int, default=1,
                        help='Number of layers in enc/dec.')
    parser.add_argument('-enc_layers', type=int, default=1,
                        help='Number of layers in the encoder')
    parser.add_argument('-dec_layers', type=int, default=1,
                        help='Number of layers in the decoder')

    parser.add_argument('-rnn_size', type=int, default=250,
                        help='Size of LSTM hidden states')

    parser.add_argument('-rnn_type', type=str, default='LSTM',
                        choices=['LSTM', 'GRU'],
                        help="""The gate type to use in the RNNs""")
    parser.add_argument('-brnn_merge', default='concat',
                        choices=['concat', 'sum'],
                        help="Merge action for the bidir hidden states")

    # Attention options
    parser.add_argument('-global_attention', type=str, default='general',
                        choices=['dot', 'general', 'mlp'],
                        help="""The attention type to use:
                        dotprot or general (Luong) or MLP (Bahdanau)""")
    parser.add_argument('-attn_hidden', type=int, default=-1,
                        help="if attn_hidden > 0, then attention score = f(Ue) B f(Ud)")

    # Layout options
    parser.add_argument('-layout_token_prune', action="store_true",
                        help="Predict whether a token appears in the layout sequence.")
    parser.add_argument('-bpe', action='store_true',
                        help="Whether use BPE.")

    # Target options
    parser.add_argument('-no_share_emb_layout_encoder', action="store_true",
                        help='Whether share embeddings for layout encoder.')
    parser.add_argument('-mask_target_loss', action="store_true",
                        help='Whether mask target sequence loss.')
    parser.add_argument('-lay_co_attention', action="store_true",
                        help='Use co-attention for layout encoder towards input.')
    parser.add_argument('-q_co_attention', action="store_true",
                        help='Use co-attention for input encoder towards layout.')

    # Ablation
    parser.add_argument('-no_lay_encoder', action="store_true",
                        help='No layout RNN encoder.')


def preprocess_opts(parser):
    parser.add_argument('-root_dir', default='',
                        help="Path to the root directory.")
    parser.add_argument('-dataset', default='atis',
                        help="Name of dataset.")
    # Dictionary Options
    parser.add_argument('-src_vocab_size', type=int, default=100000,
                        help="Size of the source vocabulary")
    parser.add_argument('-src_words_min_frequency', type=int, default=0)
    parser.add_argument('-bpe_num_merge', type=int, default=30,
                        help="BPE: number of merge.")
    parser.add_argument('-bpe_min_freq', type=int, default=0,
                        help="BPE: minimal bpe rule frequence.")

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
    parser.add_argument('-permute_order', type=int, default=0,
                        help="Permute order for logical forms for data augumentation.")
    parser.add_argument('-lower', action='store_true', help='lowercase data')


def train_opts(parser):
    # Model loading/saving options
    parser.add_argument('-root_dir', default='',
                        help="Path to the root directory.")
    parser.add_argument('-dataset', default='atis',
                        help="Name of dataset.")
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
    parser.add_argument('-update_word_vecs_after', type=int, default=0,
                        help='When fix_word_vecs=True, only update word vectors after update_word_vecs_after epochs.')

    # Optimization options
    parser.add_argument('-batch_size', type=int, default=64,
                        help='Maximum batch size')
    parser.add_argument('-epochs', type=int, default=30,
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
    parser.add_argument('-dropout_i', type=float, default=0.5,
                        help="Dropout rate (for RNN input).")
    parser.add_argument('-lock_dropout', action='store_true',
                        help="Use the same dropout mask for RNNs.")
    parser.add_argument('-weight_dropout', type=float, default=0,
                        help=">0: Weight dropout probability; applied in LSTM stacks.")
    parser.add_argument('-dropword_enc', type=float, default=0,
                        help="Drop word rate.")
    parser.add_argument('-dropword_dec', type=float, default=0,
                        help="Drop word rate.")
    parser.add_argument('-smooth_eps', type=float, default=0.1,
                        help="Label smoothing")
    parser.add_argument('-moving_avg', type=float, default=0,
                        help="Exponential moving average")
    # learning rate
    parser.add_argument('-learning_rate', type=float, default=0.005,
                        help="""Starting learning rate.""")
    parser.add_argument('-alpha', type=float, default=0.95,
                        help="Optimization hyperparameter")
    parser.add_argument('-learning_rate_decay', type=float, default=0.985,
                        help="""If update_learning_rate, decay learning rate by this much if (i) perplexity does not decrease on the validation set or (ii) epoch has gone past start_decay_at""")
    parser.add_argument('-start_decay_at', type=int, default=8,
                        help="""Start decaying every epoch after and including this epoch""")
    parser.add_argument('-start_checkpoint_at', type=int, default=30,
                        help="""Start checkpointing every epoch after and including this epoch""")

    parser.add_argument('-report_every', type=int, default=50,
                        help="Print stats at this interval.")
    parser.add_argument('-exp', type=str, default="",
                        help="Name of the experiment for logging.")

    # loss
    parser.add_argument('-coverage_loss', type=float, default=0,
                        help="Attention coverage loss.")


def translate_opts(parser):
    parser.add_argument('-root_dir', default='',
                        help="Path to the root directory.")
    parser.add_argument('-dataset', default='atis',
                        help="Name of dataset.")
    parser.add_argument('-model_path', required=True,
                        help='Path to model .pt file')
    parser.add_argument('-split', default="dev",
                        help="Path to the evaluation annotated data")
    parser.add_argument('-output', default='pred.txt',
                        help="""Path to output the predictions (each line will be the decoded sequence""")
    parser.add_argument('-run_from', type=int, default=0,
                        help='Only evaluate run.* >= run_from.')
    parser.add_argument('-batch_size', type=int, default=500,
                        help='Batch size')
    parser.add_argument('-beam_size', type=int, default=0,
                        help='Beam size')
    parser.add_argument('-n_best', type=int, default=1,
                        help='N-best size')
    parser.add_argument('-max_lay_len', type=int, default=50,
                        help='Maximum layout decoding length.')
    parser.add_argument('-max_tgt_len', type=int, default=100,
                        help='Maximum layout decoding length.')
    parser.add_argument('-gpu', type=int, default=0,
                        help="Device to run on")
    parser.add_argument('-gold_layout', action='store_true',
                        help="Given the golden layout sequences for evaluation.")
    parser.add_argument('-attn_ignore_small', type=float, default=0,
                        help="Ignore small attention scores.")
