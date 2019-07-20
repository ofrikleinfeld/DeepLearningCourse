import numpy as np

import load_data
import networks
import optimizers


if __name__ == '__main__':

    # train_mean, train_std, train_data, train_labels = load_data.load_dataset("../../data/train.csv", normalize=True)
    # validation_data, validation_labels = load_data.load_dataset("../../data/validate.csv", normalize=(train_mean, train_std))
    #
    # load_data.save_data_as_pickle_gz((train_data, train_labels, validation_data, validation_labels),
    #                                  file_name="data/training_validation_normalized.pkl.gz")

    train_data, train_labels, validation_data, validation_labels = load_data.load_training_validation_data(
                                                                    "data/training_validation_normalized.pkl.gz")

    train_data = train_data.reshape(-1, 3, 32, 32)
    #train_data = train_data.reshape(-1, 3072)
    train_data = train_data
    train_labels = train_labels
    train_length = len(train_data)
    validation_data = validation_data.reshape(-1, 3, 32, 32)
    #validation_data = validation_data.reshape(-1, 3072)
    valid_length = len(validation_data)

    model = networks.SimplerCNN()
    optimizer = optimizers.SGDOptimizer(model, lr=0.01)
    num_epochs = 100
    batch_size = 1
    batch_indices = range(0, round((len(train_data)/batch_size)) * batch_size, batch_size)

    for epoch in range(num_epochs):
        epoch_loss = 0
        model.set_mode(('train'))
        train_data, train_labels = load_data.shuffle_batch(train_data, train_labels )
        for k in batch_indices:
            x_batch = train_data[k: k + batch_size]
            y_batch = train_labels[k: k + batch_size]

            output = model(x_batch)
            loss_tmp = np.sum(np.sum(-np.log(output) * y_batch, axis=1))
            if np.isnan(loss_tmp):
                loss = np.sum(np.sum(-np.log(output+1e-15) * y_batch, axis=1))
            else:
                loss = loss_tmp

            model.backward(y_batch)
            optimizer.make_step()


            epoch_loss += loss

        print(f"Epoch {epoch + 1} - average loss is: {epoch_loss / train_length}")
        model.set_mode('test')
        # evaluate accuracy on training set
        num_correct = 0
        for k in batch_indices:
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
        sample_size = 10000
        random_indices = np.random.choice(valid_length, sample_size)
        validation_sample = validation_data[random_indices, :]
        validation_sample_labels = validation_labels[random_indices]

        # generate batches predict and compare
        validation_batch_indices = range(0, round((sample_size/batch_size)) * batch_size, batch_size)

        num_correct = 0
        for k in validation_batch_indices:
            x_batch = validation_sample[k: k + batch_size]
            y_batch = validation_sample_labels[k: k + batch_size]

            output = model(x_batch)
            predictions = np.argmax(output, axis=1)
            correct_labels = np.argmax(y_batch, axis=1)
            batch_correct_predictions = np.sum(predictions
                                               == correct_labels)
            num_correct += batch_correct_predictions
        print(num_correct, sample_size)
        print(f"Epoch {epoch + 1} - prediction accuracy on validation set is: {num_correct / sample_size}")
