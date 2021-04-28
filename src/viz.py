from utils import *
from train import *
from network import *
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA

def twod_viz(args, model=None, num_points=5):
    device = init_device()
    test_dataset = get_dataset('test.csv')
    test_dataloader = DataLoader(test_dataset,
            batch_size=num_points,
            collate_fn=collate_fn_transformer,
            num_workers=1)
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
    if args.load_path is not None:
        if os.path.isfile(args.load_path):
            _, best, model, _ = load_ckp(args.load_path, model, optimizer)
        else:
            print(f"[WARN] Could not find checkpoint '{args.load_path}'.")
            print(f"[WARN] Training from initial model...")
    model.eval()

    for batch in test_dataloader:
        batch = process_batch(batch)
        x, _ = batch
        text, mel, text_len, mel_len = x

        # Get the last latent state of sequence for both text and speech
        text_latent, _ = model.text_m.encode(text, text_len)[:, -1, :].numpy()
        speech_latent, _ = model.speech_m.encode(mel, mel_len)[:, -1, :].numpy()
        combined = torch.cat([text_latent, speech_latent], dim=0)
        break

    latents = StandardScaler.fit_transform(combined)
    pca = PCA(n_components=2)
    principalComponents = pca.fit_transform(latents)
    # We should now have a 2 * num_points x 2 array


   
    fig = plt.figure()

    for idx, cl in enumerate(np.unique(labels)):
        x = [principalComponents[idx, 0], principalComponents[idx + num_points, 0]]
        y = [principalComponents[idx, 1], principalComponents[idx + num_points, 1]]
        plt.scatter(x=x,
                    y=y,
                    label=idx)

    plt.legend(loc='upper left')
    plt.xlim(-2.75, 2.75)
    plt.ylim(-2.75, 2.75)
    plt.xlabel("Dim 1")
    plt.ylabel("Dim 2")
    plt.title("TITLE GOES HERE")
    plt.show()
    fig.savefig('latent_2d_viz.png')