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
    # train_data = train_data[:4000]
    train_length = len(train_data)
    validation_data = validation_data.reshape(-1, 3, 32, 32)
    # validation_data = validation_data.reshape(-1, 3072)
    valid_length = len(validation_data)

    model = networks.SimpleCNN()
    optimizer = optimizers.SGDOptimizer(model, lr=0.3, weights_decay='L2', weights_decay_rate=0.001)
    num_epochs = 50
    batch_size = 32

    validation_sample_size = 1000
    train_batch_indices = range(0, round((len(train_data) / batch_size)) * batch_size, batch_size)
    validation_batch_indices = range(0, round((validation_sample_size / batch_size)) * batch_size, batch_size)
    best_validation_accuracy = 0

    for epoch in range(num_epochs):
        epoch_loss = 0

        for k in train_batch_indices:
            x_batch = train_data[k: k + batch_size]
            y_batch = train_labels[k: k + batch_size]

            x_batch, y_batch = load_data.shuffle_batch(x_batch, y_batch)
            output = model(x_batch)
            loss = np.sum(np.sum(-np.log(output+1e-15) * y_batch, axis=1))

            model.backward(y_batch)
            optimizer.make_step()

            epoch_loss += loss

        print(f"Epoch {epoch + 1} - average loss is: {epoch_loss / train_length}")

        # evaluate accuracy on training set
        num_correct = 0
        for k in train_batch_indices:
            x_batch = train_data[k: k + batch_size]
            y_batch = train_labels[k: k + batch_size]

            output = model(x_batch)
            predictions = np.argmax(output, axis=1)
            correct_labels = np.argmax(y_batch, axis=1)
            batch_correct_predictions = np.sum(predictions == correct_labels)
            num_correct += batch_correct_predictions

        print(f"Epoch {epoch + 1} - prediction accuracy on training set is: {num_correct / train_length}")

        # evaluate accuracy on validation
        # choose randomly 1000 samples from validation set
        random_indices = np.random.choice(valid_length, validation_sample_size)
        validation_sample = validation_data[random_indices, :, :, :]
        validation_sample_labels = validation_labels[random_indices]

        # generate batches predict and compare
        num_correct = 0
        for k in validation_batch_indices:
            x_batch = validation_sample[k: k + batch_size]
            y_batch = validation_sample_labels[k: k + batch_size]

            output = model(x_batch)
            predictions = np.argmax(output, axis=1)
            correct_labels = np.argmax(y_batch, axis=1)
            batch_correct_predictions = np.sum(predictions == correct_labels)
            num_correct += batch_correct_predictions

        validation_accuracy = num_correct / validation_sample_size
        if validation_accuracy > best_validation_accuracy:
            load_data.save_data_as_pickle_gz(model, "cnn_lr0.3_all.txt")
            best_validation_accuracy = validation_accuracy

        print(f"Epoch {epoch + 1} - prediction accuracy on validation set is: {validation_accuracy}")
