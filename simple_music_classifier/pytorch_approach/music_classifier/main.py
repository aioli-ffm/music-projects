#!/usr/bin/python
import data
import graph
import torch
from torch import optim, nn
import numpy as np

is_gpu = torch.cuda.is_available()

if __name__ == "__main__":
    CHUNK_SIZE = 100000
    BATCH_SIZE = None
    LEARNING_RATE = 1e-3

    iterations = 100000

    cv_frequency = 100

    categories = data.generate_categories("/data/gtzan")
    CLASSES = categories.keys() # ["rock", "blues", ...]
    model = graph.mlp_def(CHUNK_SIZE, len(CLASSES), 300, 100)
    optimizer = optim.Adam(model.parameters(), LEARNING_RATE, weight_decay=1e-4)
    loss_fn = nn.CrossEntropyLoss()

    if is_gpu:
        model = model.cuda()
        loss_fn = loss_fn.cuda()

    losses = []

    from sklearn.metrics import confusion_matrix

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

        y_test = target_tensor.cpu().data.numpy()
        y_pred_all = output.cpu().data.numpy()
        y_pred = np.argmax(y_pred_all, axis=1)


        cnf_matrix = confusion_matrix(y_test, y_pred)
        print loss
        print cnf_matrix

        #_, bla = torch.max(output.data, 1)
        #print(loss.data)
        #print(bla, target_tensor.data)
        """
        uargh = 0
        for pred in output.data:
            conf_matrix.add_prediction(pred, sample_cats[uargh])
            uargh += 1

        if i % cv_frequency == cv_frequency - 1:
            print conf_matrix
        """
