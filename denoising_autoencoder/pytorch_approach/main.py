#!/usr/bin/python
import data
import graph
import torch
from torch import optim, nn
import numpy as np
from sklearn.metrics import confusion_matrix

if __name__ == "__main__":
    CHUNK_SIZE = 20000
    BATCH_SIZE = 512
    LEARNING_RATE = 1e-3
    iterations = 100000

    is_gpu = torch.cuda.is_available()

    categories = data.generate_categories("/data/gtzan")
    CLASSES = categories.keys() # ["rock", "blues", ...]
    model = graph.mlp_def(CHUNK_SIZE, len(CLASSES), 256, 128)
    optimizer = optim.Adam(model.parameters(), LEARNING_RATE, weight_decay=1e-4)
    loss_fn = nn.CrossEntropyLoss()

    if is_gpu:
        model = model.cuda()
        loss_fn = loss_fn.cuda()

    losses = []

    model.train()

    for i in xrange(iterations):

        target_tensor, sample_tensor = data.random_sample(categories, slice_size=CHUNK_SIZE, batch_size=BATCH_SIZE)

        if is_gpu:
            target_tensor = target_tensor.cuda()
            sample_tensor = sample_tensor.cuda()

        input_var = torch.autograd.Variable(sample_tensor)
        target_var = torch.autograd.Variable(target_tensor)

        optimizer.zero_grad()
        output = model.forward(input_var)
        loss = loss_fn(output, target_var)

        loss.backward()
        optimizer.step()

        correct = 0
        total = 0

        if i % 5 == 0:
            print "============== Epoch ", i
            y_test = target_var.cpu().data.numpy()
            y_pred_all = output.cpu().data.numpy()
            y_pred = np.argmax(y_pred_all, axis=1)

            correct += (y_pred == y_test).sum()
            total += y_pred.shape[0]

            cnf_matrix = confusion_matrix(y_test, y_pred)
            print "Loss: ", loss.data[0], ", Acc: ", (correct / float(total))
            print cnf_matrix
