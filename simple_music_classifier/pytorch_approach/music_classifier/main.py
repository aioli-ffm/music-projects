import data
import graph
import train

from torch import optim, nn


if __name__ == "__main__":

    iterations = 100000

    cv_frequency = 100

    categories = data.generate_categories("/home/str/@wasteland/ml/genres")
    optimizer = optim.Adam(graph.mlp.parameters())
    loss_fn = nn.CrossEntropyLoss()

    losses = []
    conf_matrix = data.ConfMatrix(categories)
    k = 0

    for i in xrange(iterations):
        category, target_tensor, sample_tensor = data.random_sample(categories)

        output, loss = train.train(
            graph.mlp,
            target_tensor, sample_tensor,
            optimizer,
            loss_fn,
            learning_rate=1e-3
        )

        losses.append(loss.data[0])
        k += 1
        conf_matrix.add_prediction(output, category[1])

        if i % cv_frequency == cv_frequency - 1:
            print conf_matrix
