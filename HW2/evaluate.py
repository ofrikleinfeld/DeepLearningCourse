import numpy as np

import load_data
import networks


def predict_model(data, model):
    predictions = []
    data_length = len(data)

    model.set_mode('test')
    for k in range(data_length):
        x = data[k].reshape(1, 3, 32, 32)
        output = model(x)
        prediction = np.argmax(output, axis=1)
        predictions.append(*prediction)

    return predictions


def evaluate_accuracy(predictions, labels):
    correct_indices = np.argmax(labels, axis=1)
    predictions = np.array(predictions)

    num_correct = np.sum(predictions == correct_indices)
    accuracy = num_correct / len(predictions)

    return accuracy


if __name__ == '__main__':

    # train_mean, train_std, train_data, train_labels = load_data.load_dataset("data/train.csv", normalize=True)
    # validation_data, validation_labels = load_data.load_dataset("data/validate.csv", normalize=(train_mean, train_std))
    # test_data, _ = load_data.load_dataset("data/test.csv", normalize=(train_mean, train_std))
    #
    # load_data.save_data_as_pickle_gz((validation_data, validation_labels), file_name="data/validation_normalized.pkl.gz")
    # load_data.save_data_as_pickle_gz(test_data, file_name="data/test_normalized.pkl.gz")

    validation_data, validation_labels = load_data.load_data_from_pickle_gz("data/validation_normalized.pkl.gz")
    test_data = load_data.load_data_from_pickle_gz("data/test_normalized.pkl.gz")

    validation_data = validation_data.reshape(-1, 3, 32, 32)
    valid_length = len(validation_data)
    test_data = test_data.reshape(-1, 3, 32, 32)
    test_length = len(test_data)

    cnn_model = networks.SimplerCNN.load_model("cnn_model_53_valid_static_seed.pkl")
    print(f"Model architecture is: {cnn_model}")

    validation_predictions = predict_model(validation_data, cnn_model)
    validation_accuracy = evaluate_accuracy(validation_predictions, validation_labels)
    print(f"prediction accuracy on validation set is: {validation_accuracy}")

    test_predictions = predict_model(test_data, cnn_model)
    test_predictions_one_based = [pred + 1 for pred in test_predictions]

    with open("output.txt", "w") as f:
        for pred in test_predictions_one_based:
            f.write(str(pred) + "\n")




