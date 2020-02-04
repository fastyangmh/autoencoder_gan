# import
from src.train import *
from src.model import *
from src.evaluation import *
from sklearn.model_selection import train_test_split
from torch.utils.data import TensorDataset, DataLoader
from sklearn.preprocessing import OneHotEncoder
import torch.optim as optim
import numpy as np

# def


if __name__ == "__main__":
    # parameters
    data_path = './data'
    batch_size = 5000
    epochs = 50
    lr = 0.001
    url = 'https://doc-14-80-docs.googleusercontent.com/docs/securesc/ha0ro937gcuc7l7deffksulhg5h7mbp1/q8k52e3qfie8lk20d225dogn90t2sdue/1580428800000/07963431181216383630/*/1AasIiFr9zAZr2_zGwONb5pQIMhWOoBSx?e=download'
    filename = 'A_Z_Handwritten_Data.csv'

    # download data
    if not isfile(join(data_path, filename)):
        print('download file from url')
        r = requests.get(url, stream=True)
        # Total size in bytes.
        total_size = int(r.headers.get('content-length', 0))
        block_size = 1024  # 1 Kibibyte
        t = tqdm(total=total_size, unit='iB', unit_scale=True)
        with open(join(data_path, filename), 'wb') as f:
            for data in r.iter_content(block_size):
                t.update(len(data))
                f.write(data)
        t.close()

    # load data
    print('load data')
    df = pd.read_csv(join(data_path, filename))
    data = df.values[:, 1:]
    label = df.values[:, 0].reshape(-1, 1)

    # preprocessing
    print('preprocessing')
    data = data/255.
    ohe = OneHotEncoder(categories='auto', sparse=False)
    label_ohe = ohe.fit_transform(label)
    x_train, x_test, y_train, y_test = train_test_split(
        data, label_ohe, test_size=0.3)
    train_set = TensorDataset(torch.from_numpy(
        x_train).float(), torch.from_numpy(y_train).float())
    train_loader = DataLoader(dataset=train_set, batch_size=batch_size,
                              shuffle=True, num_workers=4, pin_memory=True)
    test_set = TensorDataset(torch.from_numpy(
        x_test).float(), torch.from_numpy(y_test).float())
    test_loader = DataLoader(dataset=test_set, num_workers=4, pin_memory=True)

    # create model
    print('create model')
    in_dim = data.shape[1]
    condition_dim = label_ohe.shape[1]
    code_dim = 2
    autoencoder = AUTOENCODER(in_dim, condition_dim, code_dim, alpha=0.2)
    optimizer = optim.Adam(autoencoder.parameters(), lr=lr)
    criterion = nn.MSELoss()
    if USE_CUDA:
        autoencoder = autoencoder.cuda()

    # train
    history = train_loop((train_loader, test_loader),
                         autoencoder, optimizer, criterion, epochs)

    # evaluation
    points = np.array([(x, y) for x in np.linspace(-1, 1, 20)
                       for y in np.linspace(-1, 1, 20)])
    condition = np.zeros((400, 26))
    idx = 25
    condition[:, idx] += 1
    decoded = decoder(autoencoder, points, condition)
