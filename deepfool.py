import numpy as np
from torch.autograd import Variable
import torch as torch
import copy
from torch.autograd.gradcheck import zero_gradients
from Data_load import load_mnist
train_loader,test_loder = load_mnist()
from LENET import  train_eval_lenet,get_accuracy
def deepfool(image, net, num_classes=10, overshoot=0.02, max_iter=50):

    """
       :param image: Image of size HxWx3
       :param net: network (input: images, output: values of activation **BEFORE** softmax).
       :param num_classes: num_classes (limits the number of classes to test against, by default = 10)
       :param overshoot: used as a termination criterion to prevent vanishing updates (default = 0.02).
       :param max_iter: maximum number of iterations for deepfool (default = 50)
       :return: minimal perturbation that fools the classifier, number of iterations that it required, new estimated_label and perturbed image
    """
    is_cuda = torch.cuda.is_available()

    if is_cuda:
        # print("Using GPU")
        image = image.cuda()
        net = net.cuda()
    else:
        pass
    # [1] use because LENET forward has 2 outputs
    f_image = net.forward(Variable(image[None, :, :,:], requires_grad=True)).data.cpu().numpy().flatten()
    I = (np.array(f_image)).flatten().argsort()[::-1]

    I = I[0:num_classes]
    label = I[0]

    input_shape = image.cpu().numpy().shape
    pert_image = copy.deepcopy(image)
    w = np.zeros(input_shape)
    r_tot = np.zeros(input_shape)

    loop_i = 0

    x = Variable(pert_image[None, :], requires_grad=True)
    # [1] use because LENET forward has 2 outputs
    fs = net.forward(x)
    fs_list = [fs[0,I[k]] for k in range(num_classes)]
    k_i = label

    while k_i == label and loop_i < max_iter:

        pert = np.inf
        fs[0, I[0]].backward(retain_graph=True)
        grad_orig = x.grad.data.cpu().numpy().copy()

        for k in range(1, num_classes):
            zero_gradients(x)

            fs[0, I[k]].backward(retain_graph=True)
            cur_grad = x.grad.data.cpu().numpy().copy()

            # set new w_k and new f_k
            w_k = cur_grad - grad_orig
            f_k = (fs[0, I[k]] - fs[0, I[0]]).data.cpu().numpy()

            pert_k = abs(f_k)/np.linalg.norm(w_k.flatten())

            # determine which w_k to use
            if pert_k < pert:
                pert = pert_k
                w = w_k

        # compute r_i and r_tot
        # Added 1e-4 for numerical stability
        r_i =  (pert+1e-4) * w / np.linalg.norm(w)
        r_tot = np.float32(r_tot + r_i)

        if is_cuda:
            pert_image = image + (1+overshoot)*torch.from_numpy(r_tot).cuda()
        else:
            pert_image = image + (1+overshoot)*torch.from_numpy(r_tot)

        x = Variable(pert_image, requires_grad=True)
        # [1] use because LENET forward has 2 outputs
        fs = net.forward(x)
        k_i = np.argmax(fs.data.cpu().numpy().flatten())

        loop_i += 1

    r_tot = (1+overshoot)*r_tot

    return r_tot, loop_i, label, k_i, pert_image
def robustness(test_loder,model):
    total = 0
    robustness=0
    for images, lables in test_loder:
        for i in range(images.size()[0]):#images.size()[0]
            total += 1
            r_tot, loop_i, label, k_i, pert_image = deepfool(images[i], model)
            robustness += (
                    np.linalg.norm(r_tot.flatten(), 2) / np.linalg.norm(images[i].numpy().flatten(), 2))
    # print("robustness_of_orginal_model is {}".format(robustness/total))
    return torch.tensor(robustness/total)
# if probability of true class is greater than sum of probablities of other classes
# the the accuracy is same and robustness decreased
# thus we find bit that f[true]-sum(f[others])>0 and mimimize
def robustness_loss(images,model) :
    total_image=0
    loss=0
    for i in range(images.size()[0]):
        total_image += 1
        f_image = model.forward(Variable(images[i][None, :, :, :])).data.cpu().numpy().flatten()
        I = (np.array(f_image)).flatten().argsort()[::-1]
        x = Variable(images[i][None, :])
        fs = model.forward(x)
        loss += (fs[0, I[0]]-(fs[0, I[1]]+fs[0, I[2]]+
                             fs[0, I[3]]+fs[0, I[4]]+fs[0, I[5]]+
                             fs[0, I[6]]+fs[0, I[7]]+fs[0, I[8]]+fs[0, I[9]]))/fs[0, I[0]]

    return  loss/total_image
def robustness_image(images,model):
    total = 0
    robustness=0
    for i in range(images.size()[0]):#images.size()[0]
        total += 1
        r_tot, loop_i, label, k_i, pert_image = deepfool(images[i], model)
        robustness += (
                np.linalg.norm(r_tot.flatten(), 2) / np.linalg.norm(images[i].numpy().flatten(), 2))
    # print("robustness_of_orginal_model is {}".format(robustness/total))
    return torch.tensor(robustness/total)


if __name__ == '__main__':
    model = train_eval_lenet(training=False)
    robustness(test_loder,model)