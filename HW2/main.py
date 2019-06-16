import numpy as np

import load_data
import networks


if __name__ == '__main__':

    # train_mean, train_std, train_data, train_labels = load_data.load_dataset("data/train.csv", normalize=True)
    # validation_data, validation_labels = load_data.load_dataset("data/validate.csv", normalize=(train_mean, train_std))
    #
    # load_data.save_data_as_pickle_gz((train_data, train_labels, validation_data, validation_labels),
    #                                  file_name="data/training_validation_normalized.pkl.gz")

    train_data, train_labels, validation_data, validation_labels = load_data.load_training_validation_data(
                                                                    "data/training_validation_normalized.pkl.gz")

    train_data = train_data.reshape(-1, 3, 32, 32)
    validation_data = validation_data.reshape(-1, 3, 32, 32)

    model = networks.CNN()
    num_epochs = 5
    batch_size = 32
    batch_indices = range(0, round((len(train_data)/batch_size)) * batch_size, batch_size)

    for epoch in range(num_epochs):
        epoch_loss = 0

        for k in batch_indices:
            x_batch = train_data[k: k + batch_size]
            y_batch = train_labels[k: k + batch_size]

            x_batch, y_batch = load_data.shuffle_batch(x_batch, y_batch)
            output = model(x_batch)
            loss = -np.log(np.sum(output * y_batch, axis=1))
            print(loss)
            print(loss.shape)
