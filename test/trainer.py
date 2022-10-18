
def train(model, dataloader, epochs, epoch_start):
    for epoch in range(epoch_start, epochs):
        model.train(epoch, dataloader)
        model.logPrint(epoch)
        model.stepSchedule(epoch)

def train(model, dataloader):
    for i, batch in enumerate(dataloader):
        model.test_on_batch(i, batch)
    model.logPrint()
