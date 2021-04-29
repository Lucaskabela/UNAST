from utils import *
from train import *
from network import *
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA

def process_batch(batch):
    # Pos_text is unused so don't even bother loading it up
    text, mel, text_len, mel_len = batch

    # Send stuff we use alot to device this is character, mel, mel_input, and pos_mel
    text, mel, = text.to(DEVICE), mel.to(DEVICE)
    text_len, mel_len = text_len.to(DEVICE), mel_len.to(DEVICE)

    return (text, mel, text_len, mel_len), (None)

def twod_viz(args, model=None, num_points=5):

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
    to_keep = 5
    # Find the 5 with the greatest seperation!
    texts = principalComponents[:num_points]
    speech = principalComponents[num_points:]
    dist = numpy.linalg.norm(texts-speech)
    indices = (-dist).argsort()[:to_keep]
    final_points = np.concatenate([texts[indices], speech[num_points + indices]])
    refit = StandardScaler().fit_transform(final_points)

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
                    label=cl,)
    i = 0
    for x, y in zip(x, y):
        plt.text(x, y, str(i % to_keep), fontsize=12)
        i += 1
    plt.legend(loc='upper right')
    plt.xlim(-2.75, 2.75)
    plt.ylim(-2.75, 2.75)
    plt.xlabel("Dim 1")
    plt.ylabel("Dim 2")
    plt.title("TITLE GOES HERE")
    plt.show()
    fig.savefig('latent_2d_viz.png')
    print("Plot accounted for ", pca.explained_variance_ratio_ )

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
