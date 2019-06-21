import numpy as np

import load_data
import networks
import optimizers


if __name__ == '__main__':

    # train_mean, train_std, train_data, train_labels = load_data.load_dataset("data/train.csv", normalize=True)
    # validation_data, validation_labels = load_data.load_dataset("data/validate.csv", normalize=(train_mean, train_std))
    #
    # load_data.save_data_as_pickle_gz((train_data, train_labels, validation_data, validation_labels),
    #                                  file_name="data/training_validation_normalized.pkl.gz")

    train_data, train_labels, validation_data, validation_labels = load_data.load_training_validation_data(
                                                                    "data/training_validation_normalized.pkl.gz")

    train_data = train_data.reshape(-1, 3, 32, 32)
    # train_data = train_data.reshape(-1, 3072)
    train_data = train_data[:4000]
    train_length = len(train_data)
    # validation_data = validation_data.reshape(-1, 3, 32, 32)
    # validation_data = validation_data.reshape(-1, 3072)
    # validation_data = validation_data[:1000]
    # valid_length = len(validation_data)

    model = networks.Small_CNN()
    optimizer = optimizers.SGDOptimizer(model, lr=0.01)
    num_epochs = 10
    batch_size = 32
    # batch_size = 1
    batch_indices = range(0, round((len(train_data)/batch_size)) * batch_size, batch_size)

    for epoch in range(num_epochs):
        epoch_loss = 0
        for k in batch_indices:
            x_batch = train_data[k: k + batch_size]
            y_batch = train_labels[k: k + batch_size]

            x_batch, y_batch = load_data.shuffle_batch(x_batch, y_batch)
            output = model(x_batch)
            loss = np.sum(np.sum(-np.log(output+1e-15) * y_batch, axis=1))

            model.backward(y_batch)
            optimizer.make_step()

            epoch_loss += loss

        print(f"Epoch {epoch + 1} average loss is: {epoch_loss / train_length}")
