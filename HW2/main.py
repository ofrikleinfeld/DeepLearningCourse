import numpy as np

import load_data
import networks
import optimizers


if __name__ == '__main__':

    np.random.seed(1312)

    # train_mean, train_std, train_data, train_labels = load_data.load_dataset("../../data/train.csv", normalize=True)
    # validation_data, validation_labels = load_data.load_dataset("../../data/validate.csv", normalize=(train_mean, train_std))
    #
    # load_data.save_data_as_pickle_gz((train_data, train_labels, validation_data, validation_labels),
    #                                  file_name="data/training_validation_normalized.pkl.gz")

    train_data, train_labels, validation_data, validation_labels = load_data.load_training_validation_data(
                                                                    "data/training_validation_normalized.pkl.gz")
    train_data = train_data.reshape(-1, 3, 32, 32)
    train_data = train_data
    train_labels = train_labels
    train_length = len(train_data)
    validation_data = validation_data.reshape(-1, 3, 32, 32)
    valid_length = len(validation_data)

    model = networks.SimplerCNN()
    print(model)
    optimizer = optimizers.SGDOptimizer(model, lr=0.1, momentum=0.6)
    lr_scheduler = optimizers.LearningRateScheduler(optimizer, decay_factor=0.95, decay_patience=5)
    num_epochs = 80
    batch_size = 32

    validation_sample_size = len(validation_data)
    train_batch_indices = range(0, round((len(train_data) / batch_size)) * batch_size, batch_size)
    validation_batch_indices = range(0, round((validation_sample_size / batch_size)) * batch_size, batch_size)
    best_validation_accuracy = 0

    for epoch in range(num_epochs):
        epoch_loss = 0
        model.set_mode('train')
        train_data, train_labels = load_data.shuffle_batch(train_data, train_labels)
        for batch_index, k in enumerate(train_batch_indices, 1):
            x_batch = train_data[k: k + batch_size]
            y_batch = train_labels[k: k + batch_size]

            output = model(x_batch)
            loss_tmp = np.average(np.sum(-np.log(output) * y_batch, axis=1))
            if np.isnan(loss_tmp):
                loss = np.average(np.sum(-np.log(output+1e-15) * y_batch, axis=1))
            else:
                loss = loss_tmp

            model.backward(y_batch)
            optimizer.make_step()

            epoch_loss += loss

            # if batch_index == 7:
            #     optimizer.get_layers_gardients()

        print(f"Epoch {epoch + 1} - average loss is: {epoch_loss / len(train_batch_indices)}")
        model.set_mode('test')
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
        sample_size = 10000
        random_indices = np.random.choice(valid_length, sample_size)
        validation_sample = validation_data[random_indices, :]
        validation_sample_labels = validation_labels[random_indices]

        # generate batches predict and compare
        validation_batch_indices = range(0, round((sample_size/batch_size)) * batch_size, batch_size)
        val_epoch_loss = 0
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
            loss_tmp = np.average(np.sum(-np.log(output) * y_batch, axis=1))

            if np.isnan(loss_tmp):
                loss = np.average(np.sum(-np.log(output+1e-15) * y_batch, axis=1))
            else:
                loss = loss_tmp

            val_epoch_loss += loss

        validation_accuracy = num_correct / valid_length
        print(f"Epoch {epoch + 1} - average validation loss is: {val_epoch_loss / len(validation_batch_indices)}")
        print(f"Epoch {epoch + 1} - prediction accuracy on validation set is: {validation_accuracy}")

        # perform learning rate updates by using scheduler
        lr_scheduler.make_step(validation_accuracy)

        # save model if validation accuracy is best so far
        if validation_accuracy > best_validation_accuracy:
            print(f"Epoch {epoch + 1} - Saving best model so far to disk")
            model.save_model()
            best_validation_accuracy = validation_accuracy
