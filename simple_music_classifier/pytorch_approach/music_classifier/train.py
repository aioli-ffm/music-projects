
def train(graph, target_tensor, sample_tensor, optimizer, loss_fn, learning_rate = 0.005):

    output = graph.forward(sample_tensor)[0]
    #print output

    loss = loss_fn(output, target_tensor)
    loss.backward()
    optimizer.step()

    return output, loss
