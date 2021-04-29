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
    model = initialize_model(args)
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
    labels = ['text', 'speech']
    for idx, cl in enumerate(np.unique(labels)):
        plt.scatter(x=x[idx * num_points : (idx + 1) * num_points ],
                    y=y[idx * num_points : (idx + 1) * num_points ],
                    label=cl,)
    i = 0
    for x, y in zip(xs, ys):
        plt.text(x, y, str(i % num_points), fontsize=12)

    plt.legend(loc='upper left')
    plt.xlim(-2.75, 2.75)
    plt.ylim(-2.75, 2.75)
    plt.xlabel("Dim 1")
    plt.ylabel("Dim 2")
    plt.title("TITLE GOES HERE")
    plt.show()
    fig.savefig('latent_2d_viz.png')


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', help='JSON config files')
    args = parse_with_config(parser)

    global DEVICE
    DEVICE = init_device(args)

    twod_viz(args)
