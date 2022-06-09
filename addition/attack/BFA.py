import copy
import random
import torch
from models.quantization import quan_Conv2d, quan_Linear, quantize
import operator
from attack.data_conversion import *
import deepfool as dpf
from main import accuracy
from utils import AverageMeter
def NOT( a ):
    return 1-a
def validate(val_loader, model, criterion,partially=False):
    use_cuda=False
    losses = AverageMeter()
    top1 = AverageMeter()
    top5 = AverageMeter()

    # switch to evaluate mode
    model.eval()
    output_summary = []  # init a list for output summary

    with torch.no_grad():
        for i, (input, target) in enumerate(val_loader):
            if use_cuda:
                target = target.cuda(non_blocking=True)
                input = input.cuda()

            # compute output
            # input = input.squeeze().view(-1, 32 * 32)  # just for FC
            output = model(input)
            loss = criterion(output, target)

            # summary the output
            # measure accuracy and record loss
            prec1, prec5 = accuracy(output.data, target, topk=(1, 5))
            losses.update(loss.item(), input.size(0))
            top1.update(prec1.item(), input.size(0))
            top5.update(prec5.item(), input.size(0))
            if partially:
                break
    return top1.avg, top5.avg, losses.avg

class BFA(object):
    def __init__(self, criterion, model, k_top=10):

        self.criterion = criterion
        self.accuracy =0
        # init a loss_dict to log the loss w.r.t each layer
        self.loss_dict = {}
        self.robustness_dict={}
        self.accuracy_dict = {}
        self.accuracy_part_dict={}
        self.robustness_loss_dict={}
        self.bit_counter = 0
        self.k_top = k_top
        self.n_bits2flip = 0
        self.loss = 0
        self.robustness=0
        self.accuracy_part=0
        # self.MGDloss =0
        # attributes for random attack
        self.module_list = []
        for name, m in model.named_modules():
            if isinstance(m, quan_Conv2d) or isinstance(m, quan_Linear):
                self.module_list.append(name)       

    def flip_bit(self, m):
        '''
        the data type of input param is 32-bit floating, then return the data should
        be in the same data_type.
        '''
        if self.k_top is None:
            k_top = m.weight.detach().flatten().__len__()
        else: 
            k_top = self.k_top
        # 1. flatten the gradient tensor to perform topk
        w_grad_topk, w_idx_topk = m.weight.grad.detach().abs().view(-1).topk(k_top,largest=True)
        # update the b_grad to its signed representation
        w_grad_topk = m.weight.grad.detach().view(-1)[w_idx_topk]

        # 2. create the b_grad matrix in shape of [N_bits, k_top]
        b_grad_topk = w_grad_topk * m.b_w.data

        # 3. generate the gradient mask to zero-out the bit-gradient
        # which can not be flipped
        b_grad_topk_sign = (b_grad_topk.sign() +
                            1) * 0.5  # zero -> negative, one -> positive
        # convert to twos complement into unsigned integer
        w_bin = int2bin(m.weight.detach().view(-1), m.N_bits).short()
        # print(w_idx_topk)
        w_bin_topk = w_bin[w_idx_topk]  # get the weights whose grads are topk
        # generate two's complement bit-map
        b_bin_topk = (w_bin_topk.repeat(m.N_bits,1) & m.b_w.abs().repeat(1,k_top).short()) \
        // m.b_w.abs().repeat(1,k_top).short()
        grad_mask = NOT(b_bin_topk ^ b_grad_topk_sign.short())

        # 4. apply the gradient mask upon ```b_grad_topk``` and in-place update it
        b_grad_topk *= grad_mask.float()

        # 5. identify the several maximum of absolute bit gradient and return the
        # index, the number of bits to flip is self.n_bits2flip
        grad_max = b_grad_topk.abs().max()
        _, b_grad_max_idx = b_grad_topk.abs().view(-1).topk(self.n_bits2flip,largest=True)
        bit2flip = b_grad_topk.clone().view(-1).zero_()

        if grad_max.item() != 0:  # ensure the max grad is not zero
            bit2flip[b_grad_max_idx] = 1
            bit2flip = bit2flip.view(b_grad_topk.size())
        else:
            pass

        # 6. Based on the identified bit indexed by ```bit2flip```, generate another
        # mask, then perform the bitwise xor operation to realize the bit-flip.
        w_bin_topk_flipped = (bit2flip.short() * m.b_w.abs().short()).sum(0, dtype=torch.int16) \
                ^ w_bin_topk

        # 7. update the weight in the original weight tensor
        w_bin[w_idx_topk] = w_bin_topk_flipped  # in-place change
        param_flipped = bin2int(w_bin,
                                m.N_bits).view(m.weight.data.size()).float()

        return param_flipped

    def progressive_bit_search(self, model, data, target,test_loader):
        ''' 
        Given the model, base on the current given data and target, go through
        all the layer and identify the bits to be flipped. 
        '''
        # Note that, attack has to be done in evaluation model due to batch-norm.
        # see: https://discuss.pytorch.org/t/what-does-model-eval-do-for-batchnorm-layer/7146
        model.eval()

        # 1. perform the inference w.r.t given data and target
        output = model(data)
        #         _, target = output.data.max(1)
        self.loss = self.criterion(output, target)
        self.robustness =dpf.robustness_image(data,model)
        self.accuracy,_,_ = validate(test_loader,model,self.criterion)
        self.robustness_loss = dpf.robustness_loss(data, model)
        self.accuracy_part = validate(test_loader,model,self.criterion,True)
        # self.MGDloss = self.loss + self.robustness_loss
        # self.robustness = dpf.robustness_image(data,model)
        # print(self.MGDloss.item())
        # 2. zero out the grads first, then get the grads
        for m in model.modules():
            if isinstance(m, quan_Conv2d) or isinstance(m, quan_Linear):
                if m.weight.grad is not None:
                    m.weight.grad.data.zero_()

        self.robustness_loss.backward()
        # init the loss_max to enable the while loop
        self.loss_max = self.loss.item()
        self.robustness_min = self.robustness.item()
        self.accuracy_max = self.accuracy
        self.robustness_loss_min= self.robustness_loss.item()
        # self.MGDloss_max = self.MGDloss.item()
        # 3. for each layer flip #bits = self.bits2flip
        print("initial robustness={}".format(self.robustness_min))
        print("initial loss={}".format(self.loss_max))
        print("initial robustness_loss={}".format(self.robustness_loss_min))
        print("initial accuracy gloabl ={}".format(self.accuracy_max))
        print("initial accuracy partial ={}".format(self.accuracy_part))
        while self.robustness_min>=self.robustness.item():

            self.n_bits2flip += 1
            # iterate all the quantized conv and linear layer
            for name, module in model.named_modules():
                if isinstance(module, quan_Conv2d) or isinstance(
                        module, quan_Linear):
                    clean_weight = module.weight.data.detach()
                    ##########################################################################3
                    attack_weight = self.flip_bit(module)
                    # change the weight to attacked weight and get loss
                    module.weight.data = attack_weight
                    output = model(data)
                    self.loss_dict[name] = self.criterion(output,
                                                          target).item()
                    self.robustness_dict[name]=dpf.robustness_image(data,model)
                    self.robustness_loss_dict[name]=dpf.robustness_loss(data, model)
                    self.accuracy_dict[name]=validate(test_loader,model,self.criterion)
                    self.accuracy_part_dict[name]=validate(test_loader,model,self.criterion,True)
                    # self.MGDloss_dict[name] = self.loss_dict[name] + self.robustness_dict[name]
                    # change the weight back to the clean weight
                    module.weight.data = clean_weight

            # after going through all the layer, now we find the layer with max loss
            max_loss_module = max(self.loss_dict.items(),
                                  key=operator.itemgetter(1))[0]
            min_robust_module = min(self.robustness_dict.items(),
                                  key=operator.itemgetter(1))[0]
            non_negative = {k: v for k, v in self.robustness_loss_dict.items() if v != 0}
            min_robustness_loss_module=min(non_negative.items(),
                                  key=operator.itemgetter(1))[0]
            max_accuracy_module = max(self.accuracy_dict.items(),
                                    key=operator.itemgetter(1))[0]
            self.loss_max = self.loss_dict[min_robustness_loss_module]
            self.robustness_min = self.robustness_dict[min_robustness_loss_module]
            self.robustness_loss_min = self.robustness_loss_dict[min_robustness_loss_module]
            self.accuracy_max = self.accuracy_dict[min_robustness_loss_module]
            self.accuracy_part_max = self.accuracy_part_dict[min_robustness_loss_module]
            print("after attack robustness={}".format(self.robustness_min))
            print("after attack loss={}".format(self.loss_max))
            print("after attack robustness loss={}".format(self.robustness_loss_min))
            print("after attack accuracy global={}".format(self.accuracy_max))
            print("after attack accuracy partial ={}".format(self.accuracy_part_max))
            # print(max_accuracy_module)
            # self.robustness_max = dpf.robustness_image(data,model)
        # 4. if the loss_max does lead to the degradation compared to the self.loss,
        # then change that layer's weight without putting back the clean weight
        for module_idx, (name, module) in enumerate(model.named_modules()):
            if name == min_robustness_loss_module:
                # print(name, self.loss.item(), loss_max)

                attack_weight = self.flip_bit(module)
                ###########################################################
                ## Attack profiling
                #############################################
                weight_mismatch = attack_weight - module.weight.detach()
                attack_weight_idx = torch.nonzero(weight_mismatch)
                print(attack_weight_idx)
                print('attacked module:', min_robustness_loss_module)
                
                attack_log = [] # init an empty list for profile
                
                for i in range(attack_weight_idx.size()[0]):
                    
                    weight_idx = attack_weight_idx[i,:].cpu().numpy()
                    weight_prior = module.weight.detach()[tuple(attack_weight_idx[i,:])].item()
                    weight_post = attack_weight[tuple(attack_weight_idx[i,:])].item()
                    
                    print('attacked weight index:', weight_idx)
                    print('weight before attack:', weight_prior)
                    print('weight after attack:', weight_post)
                    
                    tmp_list = [module_idx, # module index in the net
                                self.bit_counter + (i+1), # current bit-flip index
                                min_robustness_loss_module, # current bit-flip module
                                weight_idx, # attacked weight index in weight tensor
                                weight_prior, # weight magnitude before attack
                                weight_post # weight magnitude after attack
                                ] 
                    attack_log.append(tmp_list)

                ###############################################################    
                
                
                module.weight.data = attack_weight

        # reset the bits2flip back to 0
        self.bit_counter += self.n_bits2flip
        self.n_bits2flip = 0

        return attack_log


    def random_flip_one_bit(self, model):
        """
        Note that, the random bit-flip may not support on binary weight quantization.
        """
        chosen_module = random.choice(self.module_list)
        for name, m in model.named_modules():
            if name == chosen_module:
                flatten_weight = m.weight.detach().view(-1)
                chosen_idx = random.choice(range(flatten_weight.__len__()))
                # convert the chosen weight to 2's complement
                bin_w = int2bin(flatten_weight[chosen_idx], m.N_bits).short()
                # randomly select one bit
                bit_idx = random.choice(range(m.N_bits))
                mask = (bin_w.clone().zero_() + 1) * (2**bit_idx)
                bin_w = bin_w ^ mask
                int_w = bin2int(bin_w, m.N_bits).float()
                
                ##############################################
                ###   attack profiling
                ###############################################
                
                weight_mismatch = flatten_weight[chosen_idx] - int_w
                attack_weight_idx = chosen_idx
                
                print('attacked module:', chosen_module)
                
                attack_log = [] # init an empty list for profile
                
                
                weight_idx = chosen_idx
                weight_prior = flatten_weight[chosen_idx]
                weight_post = int_w

                print('attacked weight index:', weight_idx)
                print('weight before attack:', weight_prior)
                print('weight after attack:', weight_post)  
                
                tmp_list = ["module_idx", # module index in the net
                            self.bit_counter + 1, # current bit-flip index
                            "loss", # current bit-flip module
                            weight_idx, # attacked weight index in weight tensor
                            weight_prior, # weight magnitude before attack
                            weight_post # weight magnitude after attack
                            ] 
                attack_log.append(tmp_list)                            
                
                self.bit_counter += 1
                #################################
                
                flatten_weight[chosen_idx] = int_w
                m.weight.data = flatten_weight.view(m.weight.data.size())
                
            
                
        return attack_log


