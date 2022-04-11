import alpha_net
import os
import torch

net = alpha_net.ChessNet()
net.train()
save_as="current_net_trained8_iter1.pth.tar"
torch.save({'state_dict': net.state_dict()}, os.path.join("./model_data/",
                                                          save_as))
