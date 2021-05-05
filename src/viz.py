from utils import *
from train import *
from network import *
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
import numpy
import pylab

def process_batch(batch):
    # Pos_text is unused so don't even bother loading it up
    text, mel, text_len, mel_len = batch

    # Send stuff we use alot to device this is character, mel, mel_input, and pos_mel
    text, mel, = text.to(DEVICE), mel.to(DEVICE)
    text_len, mel_len = text_len.to(DEVICE), mel_len.to(DEVICE)

    return (text, mel, text_len, mel_len), (None)

def label_batch(args, model=None, num_points=300):
    
    test_dataset = get_dataset('test.csv')
    test_dataloader = DataLoader(test_dataset,
            batch_size=num_points,
            collate_fn=collate_fn_transformer,
            num_workers=1)
    _, _, model, _, _ = initialize_model(args)
    model.eval()

    for batch in test_dataloader:
        batch = process_batch(batch)
        x, _ = batch
        text, mel, text_len, mel_len = x

        # Get the last latent state of sequence for both text and speech
        with torch.no_grad():
            text_latent, _ = model.text_m.encode(text, text_len)
            speech_latent, _ = model.speech_m.encode(mel, mel_len)

        if args.model_type == 'rnn':
            _, t_out = text_latent
            _, s_out = speech_latent
        else:
            t_out = text_latent
            s_out = speech_latent        
        text_latent = t_out[:, -1, :]
        speech_latent = s_out[:, -1, :]
        combined = torch.cat([text_latent, speech_latent], dim=0).cpu().numpy()
        break
    label = ['text'] * num_points + ['Speech'] * num_points
    return combined, labeled

def twod_viz(args, model=None, num_points=300, to_keep=5):

    test_dataset = get_dataset('test.csv')
    test_dataloader = DataLoader(test_dataset,
            batch_size=num_points,
            collate_fn=collate_fn_transformer,
            num_workers=1)
    _, _, model, _, _ = initialize_model(args)
    model.eval()

    for batch in test_dataloader:
        batch = process_batch(batch)
        x, _ = batch
        text, mel, text_len, mel_len = x

        # Get the last latent state of sequence for both text and speech
        with torch.no_grad():
            text_latent, _ = model.text_m.encode(text, text_len)
            speech_latent, _ = model.speech_m.encode(mel, mel_len)

        if args.model_type == 'rnn':
            _, t_out = text_latent
            _, s_out = speech_latent
        else:
            t_out = text_latent
            s_out = speech_latent        
        text_latent = t_out[:, -1, :]
        speech_latent = s_out[:, -1, :]
        combined = torch.cat([text_latent, speech_latent], dim=0).cpu().numpy()
        break
    print(combined.shape)
    latents = StandardScaler().fit_transform(combined)
    pca = PCA(n_components=2)
    principalComponents = pca.fit_transform(latents)

    # Refit so we can visualize
    print(principalComponents)
    # Find the 5 with the greatest seperation!
    texts = principalComponents[:num_points]
    speech = principalComponents[num_points:]
    dist = numpy.sqrt(numpy.sum((texts-speech)**2, axis=1))
    print(dist)
    indices = (-dist).argsort()[:to_keep]
    print(indices)
    final_points = np.concatenate([texts[indices], speech[indices]])
    print(final_points.shape)
    refit = StandardScaler().fit_transform(final_points)
    # Note at this point there might be too much clustering, due to huge reduction - so add noise!
    noise = np.random.normal(0, .1, refit.shape)
    refit = refit + noise
    # We should now have a 2 * num_points x 2 array


    x = refit[:, 0]
    y = refit[:, 1]
    print(x)
    print(y)
    fig = plt.figure()
    labels = ['text', 'speech']
    for idx, cl in enumerate(np.unique(labels)):
        plt.scatter(x=x[idx * to_keep : (idx + 1) * to_keep ],
                    y=y[idx * to_keep : (idx + 1) * to_keep ],
                    label=cl,
                    marker='x')
    i = 0
    for x, y in zip(x, y):
        plt.text(x, y, str(i % to_keep), fontsize=12)
        i += 1
    plt.legend(loc='upper right')
    # plt.xlim(-2.75, 2.75)
    # plt.ylim(-2.75, 2.75)
    plt.xlabel("Dim 1")
    plt.ylabel("Dim 2")
    plt.title("TITLE GOES HERE")
    plt.show()
    fig.savefig('latent_2d_viz.png')
    print("Plot accounted for ", pca.explained_variance_ratio_ )

def Hbeta(D=np.array([]), beta=1.0):
    """
        Compute the perplexity and the P-row for a specific value of the
        precision of a Gaussian distribution.
    """

    # Compute P-row and corresponding perplexity
    P = np.exp(-D.copy() * beta)
    sumP = sum(P)
    H = np.log(sumP) + beta * np.sum(D * P) / sumP
    P = P / sumP
    return H, P


def x2p(X=np.array([]), tol=1e-5, perplexity=30.0):
    """
        Performs a binary search to get P-values in such a way that each
        conditional Gaussian has the same perplexity.
    """

    # Initialize some variables
    print("Computing pairwise distances...")
    (n, d) = X.shape
    sum_X = np.sum(np.square(X), 1)
    D = np.add(np.add(-2 * np.dot(X, X.T), sum_X).T, sum_X)
    P = np.zeros((n, n))
    beta = np.ones((n, 1))
    logU = np.log(perplexity)

    # Loop over all datapoints
    for i in range(n):

        # Print progress
        if i % 500 == 0:
            print("Computing P-values for point %d of %d..." % (i, n))

        # Compute the Gaussian kernel and entropy for the current precision
        betamin = -np.inf
        betamax = np.inf
        Di = D[i, np.concatenate((np.r_[0:i], np.r_[i+1:n]))]
        (H, thisP) = Hbeta(Di, beta[i])

        # Evaluate whether the perplexity is within tolerance
        Hdiff = H - logU
        tries = 0
        while np.abs(Hdiff) > tol and tries < 50:

            # If not, increase or decrease precision
            if Hdiff > 0:
                betamin = beta[i].copy()
                if betamax == np.inf or betamax == -np.inf:
                    beta[i] = beta[i] * 2.
                else:
                    beta[i] = (beta[i] + betamax) / 2.
            else:
                betamax = beta[i].copy()
                if betamin == np.inf or betamin == -np.inf:
                    beta[i] = beta[i] / 2.
                else:
                    beta[i] = (beta[i] + betamin) / 2.

            # Recompute the values
            (H, thisP) = Hbeta(Di, beta[i])
            Hdiff = H - logU
            tries += 1

        # Set the final row of P
        P[i, np.concatenate((np.r_[0:i], np.r_[i+1:n]))] = thisP

    # Return final P-matrix
    print("Mean value of sigma: %f" % np.mean(np.sqrt(1 / beta)))
    return P


def pca(X=np.array([]), no_dims=50):
    """
        Runs PCA on the NxD array X in order to reduce its dimensionality to
        no_dims dimensions.
    """

    print("Preprocessing the data using PCA...")
    (n, d) = X.shape
    X = X - np.tile(np.mean(X, 0), (n, 1))
    (l, M) = np.linalg.eig(np.dot(X.T, X))
    Y = np.dot(X, M[:, 0:no_dims])
    return Y


def tsne(X=np.array([]), no_dims=2, initial_dims=50, perplexity=30.0):
    """
        Runs t-SNE on the dataset in the NxD array X to reduce its
        dimensionality to no_dims dimensions. The syntaxis of the function is
        `Y = tsne.tsne(X, no_dims, perplexity), where X is an NxD NumPy array.
    """

    # Check inputs
    if isinstance(no_dims, float):
        print("Error: array X should have type float.")
        return -1
    if round(no_dims) != no_dims:
        print("Error: number of dimensions should be an integer.")
        return -1

    # Initialize variables
    X = pca(X, initial_dims).real
    (n, d) = X.shape
    max_iter = 1000
    initial_momentum = 0.5
    final_momentum = 0.8
    eta = 500
    min_gain = 0.01
    Y = np.random.randn(n, no_dims)
    dY = np.zeros((n, no_dims))
    iY = np.zeros((n, no_dims))
    gains = np.ones((n, no_dims))

    # Compute P-values
    P = x2p(X, 1e-5, perplexity)
    P = P + np.transpose(P)
    P = P / np.sum(P)
    P = P * 4.									# early exaggeration
    P = np.maximum(P, 1e-12)

    # Run iterations
    for iter in range(max_iter):

        # Compute pairwise affinities
        sum_Y = np.sum(np.square(Y), 1)
        num = -2. * np.dot(Y, Y.T)
        num = 1. / (1. + np.add(np.add(num, sum_Y).T, sum_Y))
        num[range(n), range(n)] = 0.
        Q = num / np.sum(num)
        Q = np.maximum(Q, 1e-12)

        # Compute gradient
        PQ = P - Q
        for i in range(n):
            dY[i, :] = np.sum(np.tile(PQ[:, i] * num[:, i], (no_dims, 1)).T * (Y[i, :] - Y), 0)

        # Perform the update
        if iter < 20:
            momentum = initial_momentum
        else:
            momentum = final_momentum
        gains = (gains + 0.2) * ((dY > 0.) != (iY > 0.)) + \
                (gains * 0.8) * ((dY > 0.) == (iY > 0.))
        gains[gains < min_gain] = min_gain
        iY = momentum * iY - eta * (gains * dY)
        Y = Y + iY
        Y = Y - np.tile(np.mean(Y, 0), (n, 1))

        # Compute current value of cost function
        if (iter + 1) % 10 == 0:
            C = np.sum(P * np.log(P / Q))
            print("Iteration %d: error is %f" % (iter + 1, C))

        # Stop lying about P-values
        if iter == 100:
            P = P / 4.

    # Return solution
    return Y

def initialize_model(args):
    """
        Using args, initialize starting epoch, best per, model, optimizer
    """
    text_m, speech_m, discriminator, teacher = None, None, None, get_teacher_ratio(args)
    if args.model_type == 'rnn':
        text_m = TextRNN(args)
        speech_m = SpeechRNN(args)
    elif args.model_type == 'transformer':
        text_m = TextTransformer(args)
        speech_m = SpeechTransformer(args)

    if args.use_discriminator:
        discriminator_in_dim = args.hidden * 2 if args.model_type == 'rnn' else args.hidden
        discriminator = LSTMDiscriminator(discriminator_in_dim, args.disc_hid, bidirectional=args.disc_bidirectional, num_layers=args.disc_num_layers)
    model = UNAST(text_m, speech_m, discriminator, teacher).to(DEVICE)

    # initialize optimizer
    optimizer = None
    if args.optim_type == 'adam':
        optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    elif args.optim_type == 'adamw':
        optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)

    # continue training if needed
    s_epoch, best = 0, 300
    if args.load_path is not None:
        if os.path.isfile(args.load_path):
            s_epoch, best, model, optimizer = load_ckp(args.load_path, model, optimizer)
        else:
            print(f"[WARN] Could not find checkpoint '{args.load_path}'.")
            print(f"[WARN] Training from initial model...")

    # initialize scheduler
    scheduler = None

    model.teacher.iter = s_epoch
    return s_epoch, best, model, optimizer, scheduler
    
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', help='JSON config files')
    args = parse_with_config(parser)

    global DEVICE
    DEVICE = init_device(args)

    twod_viz(args)

    X, labels = label_batch(args)
    X = x.numpy()

    Y = tsne(X, 2, initial_dims=X.shape[1], perplexity=30.0)
    pylab.scatter(Y[:, 0], Y[:, 1], 20, labels)
    pylab.show()
