import data
import graph
import torch
from torch import optim, nn

is_gpu = torch.cuda.is_available()

if __name__ == "__main__":
    CHUNK_SIZE = 100000
    BATCH_SIZE = None
    LEARNING_RATE = 1e-3

    iterations = 100000

    cv_frequency = 100

    categories = data.generate_categories("../../../datasets/gtzan")
    CLASSES = categories.keys() # ["rock", "blues", ...]
    model = graph.mlp_def(CHUNK_SIZE, len(CLASSES), 300, 100)
    optimizer = optim.Adam(model.parameters(), LEARNING_RATE, weight_decay=1e-3)
    loss_fn = nn.CrossEntropyLoss()

    if is_gpu:
        model = model.cuda()
        loss_fn = loss_fn.cuda()

    losses = []
    conf_matrix = data.ConfMatrix(categories)

    for i in xrange(iterations):

        sample_cats, target_tensor, sample_tensor = data.random_sample(categories, slice_size=CHUNK_SIZE)

        if is_gpu:
            target_tensor = target_tensor.cuda()
            sample_tensor = sample_tensor.cuda()

        sample_tensor.data = torch.squeeze(sample_tensor.data)

        output = model.forward(sample_tensor)
        loss = loss_fn(output, target_tensor)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        _, bla = torch.max(output.data, 1)
        print(loss.data)
        print(bla, target_tensor.data)
        """
        uargh = 0
        for pred in output.data:
            conf_matrix.add_prediction(pred, sample_cats[uargh])
            uargh += 1

        if i % cv_frequency == cv_frequency - 1:
            print conf_matrix
        """