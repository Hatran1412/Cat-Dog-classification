from src.trainer import train, model1
from src.tester import test



def work1():
    print('Training a model 1')
    num_epochs = 5
    for epoch in range(1, num_epochs+1):
        train(epoch, model1)
        test(model1)


if __name__ == "__main__":
    work1()
