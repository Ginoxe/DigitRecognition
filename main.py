import torch
import numpy as np
import torchvision
import matplotlib.pyplot as plt
from time import time
from torchvision import datasets, transforms
from torch import nn, optim

transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.5,), (0.5,)),])

trainset = datasets.MNIST('C:\Develop\CNN Handwritten Number\\train', download = True, train = True, transform = transform)
valset = datasets.MNIST('C:\Develop\CNN Handwritten Number\\val', download = True, train = False, transform = transform)
trainloader = torch.utils.data.DataLoader(trainset, batch_size = 64, shuffle = True)
valloader = torch.utils.data.DataLoader(valset, batch_size = 64, shuffle = True)

# dataiter = iter(trainloader)
# images, labels = next(dataiter)

# print(images.shape)
# print(labels.shape)

#----------------- 사진 보기 -------------------
# figure = plt.figure()
# num_of_images = 60
# for index in range(1, num_of_images + ex)
#     plt.axis('off')1):
#     plt.subplot(6, 10, ind
#     plt.imshow(images[index].numpy().squeeze(), cmap='gray_r')
# plt.show()

input_size = 784
hidden_sizes = [128, 64]
output_size = 10

'''
1. Sigmoid
2. Tanh
3. ReLU
4. LeakyReLU
5. PReLU
6. ELU
7. SELU
8. SiLU
'''

functions = [nn.ReLU(), nn.Tanh(), nn.Sigmoid(), nn.LeakyReLU(), nn.PReLU(), nn.ELU(), nn.SELU(), nn.SiLU()]

for function in functions:
    for i in range(2):
        model = nn.Sequential(nn.Linear(input_size, hidden_sizes[0]), function,
                            nn.Linear(hidden_sizes[0], hidden_sizes[1]), function,
                            nn.Linear(hidden_sizes[1], output_size), nn.LogSoftmax(dim = 1))

        #print(model)

        criterion = nn.NLLLoss()
        images, labels = next(iter(trainloader))
        images = images.view(images.shape[0], -1)

        logps = model(images) #log probabilities
        loss = criterion(logps, labels) #calculate the NLL loss

        #print('Before backward pass: \n', model[0].weight.grad)
        loss.backward()
        #print('After backward pass: \n', model[0].weight.grad)

        optimizer = optim.SGD(model.parameters(), lr=0.003, momentum=0.9)
        time0 = time()
        epochs = 15
        for e in range(epochs):
            running_loss = 0
            for images, labels in trainloader:
                # Flatten MNIST images into a 784 long vector
                images = images.view(images.shape[0], -1)
            
                # Training pass
                optimizer.zero_grad()
                
                output = model(images)
                loss = criterion(output, labels)
                
                #This is where the model learns by backpropagating
                loss.backward()
                
                #And optimizes its weights here
                optimizer.step()
                
                running_loss += loss.item()
            # else:
            #     print("Epoch {} - Training loss: {}".format(e, running_loss/len(trainloader)))
        print(f"------------------{function}:{i+1}------------------------")
        print("\nTraining Time (in minutes) =",(time()-time0)/60)

        # ------------------ Save Model ---------------------
        # torch.save(model, './model.pt')

        # model = torch.load('./model.pt')


        def view_classify(img, ps):
            ''' Function for viewing an image and it's predicted classes.
            '''
            ps = ps.data.numpy().squeeze()

            fig, (ax1, ax2) = plt.subplots(figsize=(6,9), ncols=2)
            ax1.imshow(img.resize_(1, 28, 28).numpy().squeeze())
            ax1.axis('off')
            ax2.barh(np.arange(10), ps)
            ax2.set_aspect(0.1)
            ax2.set_yticks(np.arange(10))
            ax2.set_yticklabels(np.arange(10))
            ax2.set_title('Class Probability')
            ax2.set_xlim(0, 1.1)
            plt.tight_layout()
            plt.show()


        # images, labels = next(iter(valloader))

        # img = images[0].view(1, 784)
        # with torch.no_grad():
        #     logps = model(img)

        # ps = torch.exp(logps)
        # probab = list(ps.numpy()[0])
        # print("Predicted Digit =", probab.index(max(probab)))
        # view_classify(img.view(1, 28, 28), ps)

        correct_count, all_count = 0, 0
        for images,labels in valloader:
            for i in range(len(labels)):
                img = images[i].view(1, 784)
                with torch.no_grad():
                    logps = model(img)

                
                ps = torch.exp(logps)
                probab = list(ps.numpy()[0])
                pred_label = probab.index(max(probab))
                true_label = labels.numpy()[i]
                if(true_label == pred_label):
                    correct_count += 1
                all_count += 1

        #print("Number Of Images Tested =", all_count)
        print("\nModel Accuracy =", (correct_count/all_count))