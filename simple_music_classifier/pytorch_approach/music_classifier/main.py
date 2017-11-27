import data
import graph
import train

from torch import optim, nn


# args.cuda = not args.no_cuda and torch.cuda.is_available()


# torch.manual_seed(args.seed)
# if args.cuda:
#     torch.cuda.manual_seed(args.seed)

        # if args.cuda:
        #     data, target = data.cuda(), target.cuda()
        # data, target = Variable(data), Variable(target)

if __name__ == "__main__":
    CHUNK_SIZE = 10000
    BATCH_SIZE = None
    LEARNING_RATE = 1e-3

    iterations = 100000

    cv_frequency = 100

    categories = data.generate_categories("../../../datasets/gtzan")
    CLASSES = categories.keys() # ["rock", "blues", ...]
    model = graph.mlp_def(CHUNK_SIZE, len(CLASSES), 128, 64)
    optimizer = optim.Adam(model.parameters(), LEARNING_RATE)
    loss_fn = nn.CrossEntropyLoss()

    losses = []
    conf_matrix = data.ConfMatrix(categories)
    k = 0

    for i in xrange(iterations):
        optimizer.zero_grad()
        category, target_tensor, sample_tensor = data.random_sample(categories, slice_size=CHUNK_SIZE)
        output = model.forward(sample_tensor)[0]
        loss = loss_fn(output, target_tensor)
        loss.backward()
        optimizer.step()

        losses.append(loss.data[0])
        k += 1
        conf_matrix.add_prediction(output, category[1])

        if i % cv_frequency == cv_frequency - 1:
            print conf_matrix
