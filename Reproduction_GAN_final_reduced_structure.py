## This verision aims to reproduce the CGAN for E2E on AWGN channel
## number of pilots set to be 0

import torch
import torch.nn as nn
from torch.autograd import Variable
import numpy as np
import matplotlib.pyplot as plt
import torch.nn.functional as F
import scipy.io as sio

# define parameters
n = 7
k = 4
M = 2**k
batch_size = 320
num_epoch = 100000
num_receiver = 1
num_transmitter = 1
num_GAN = 1
num_train = 10000
num_test = 100000
num_plot = 200
LR_t = 0.001 
LR_r = 0.001
LR_g = 0.0001
z_dimension = n
num_pilots = 0
SNR_dB = 0

SNR = 10**(SNR_dB/10)


# calculate the BER
def BER(X, Y_pre, num):
    cnt = 0
    for i in range(num):
        if X[i] == Y_pre[i]:
            cnt += 1
    BER_result = cnt/num
    return (1-BER_result)


# channel --AWGN
def channel_AWGN(x, k, n, SNR, batch_size):
    noise = np.random.randn(batch_size, n)*np.sqrt(1/(2*SNR*k/n))
    return noise + x

def channel_AWGN_fake(x, k, n, SNR, batch_size):
    noise = np.random.randn(batch_size, n)*np.sqrt(1/(2*SNR*k/n))
    return x

# traditional batch selection method
def data_generator(M, num):
    message_idx = np.random.randint(0, M, size=num)
    X = convert_to_onehot(M, num, message_idx)
    Y = X
    return message_idx, X, Y
def convert_to_onehot(M, num, idx):
    X = np.zeros([num,M],dtype=np.int32)
    for i in range(num):
        X[i,idx[i]] = 1
    return X
# idx, x, y = data_generator(3, 20)
# print('idx: ', idx, 'one-hot: ', x)



# define the whole network for training process
# input M+num_pilots*n+z_dimension      num_pilots=0
class net(nn.Module):
    def __init__(self, M, n, z_dimension):
        super(net, self).__init__()
        self.t = nn.Sequential(
            nn.Linear(M, 32),
            nn.ReLU(),
            nn.Linear(32, n)
        )
        self.g = nn.Sequential(
            nn.Linear(z_dimension+n, 64),
            nn.ReLU(),
            nn.Linear(64, 64),
            nn.ReLU(),
            nn.Linear(64, n)
            # nn.Tanh()           ## need further confirmation
        )
        self.r = nn.Sequential(
            nn.Linear(n, 32),
            nn.ReLU(),
            nn.Linear(32, M)
            # nn.Sigmoid()
        )
        
    def forward(self, x):
        temp = x
        noise = temp[:,0:z_dimension]     #input batch_size*(M+z_dimension)
        x = temp[:,z_dimension:]        
        x = self.t(x)
        x = nn.functional.normalize(x, 2, 1)*np.sqrt(n)
        x = torch.cat([noise,x],1)
        x = self.g(x)
        x = self.r(x)
        # x = F.log_softmax(x,dim=1)
        return x

    def encoding(self, x):
        x = self.t(x)
        x = nn.functional.normalize(x, 2, 1)*np.sqrt(n)
        return x


# define transmitter
class transmitter(nn.Module):
    def __init__(self, M, n):
        super(transmitter, self).__init__()
        self.t = nn.Sequential(
            nn.Linear(M, 32),
            nn.ReLU(),
            nn.Linear(32, n),
        )
    def forward(self, x):
        x = self.t(x)
        x = nn.functional.normalize(x, 2, 1)*np.sqrt(n)
        return x


# define receiver
# input num_pilots*n + n  num_pilots=0
class receiver(nn.Module):
    def __init__(self, M, n):
        super(receiver, self).__init__()
        self.r = nn.Sequential(
            nn.Linear(n, 32),
            nn.ReLU(),
            nn.Linear(32, M)
            # nn.Sigmoid()
        )
    def forward(self, x):
        x = self.r(x)
        # x = F.log_softmax(x,dim=1)
        return x

# define generator of GAN
class generator(nn.Module):
    def __init__(self, M, n, z_dimension):
        super(generator, self).__init__()
        self.g = nn.Sequential(
            nn.Linear(z_dimension+n, 64),
            nn.ReLU(),
            nn.Linear(64, 64),
            nn.ReLU(),
            nn.Linear(64, n),
            # nn.Tanh()           ## need further confirmation
        )
    def forward(self, x):
        x = self.g(x)
        return x
 
# define discriminator of GAN
class discriminator(nn.Module):
    def __init__(self, M, n):
        super(discriminator, self).__init__()
        self.d = nn.Sequential(
            nn.Linear(n+n, 32),
            nn.ReLU(),
            nn.Linear(32, 32),
            nn.ReLU(),
            nn.Linear(32, 1),
            nn.Sigmoid()
        )
    def forward(self, x):
        x = self.d(x)
        return x

def test(M, n, k, x, y, num, SNR, NT, T, R):
    x_test = Variable(torch.Tensor(x))
    y_test = Variable(torch.Tensor(y))
    # update the parameters
    R_dict = R.state_dict()
    T_dict = T.state_dict()
    NT_dict = NT.state_dict()
    R_dict_new = {k:v for k,v in NT_dict.items() if k in R_dict}
    T_dict_new = {k:v for k,v in NT_dict.items() if k in T_dict}
    R_dict.update(R_dict_new)
    T_dict.update(T_dict_new)
    R.load_state_dict(R_dict)
    T.load_state_dict(T_dict)
    # get the results over real channel
    out_t = T(x_test)
    out_t_np = out_t.data.numpy()
    # out_channel = channel_AWGN_fake(out_t_np, k, n, SNR, num)
    out_channel = channel_AWGN(out_t_np, k, n, SNR, num)
    in_r = Variable(torch.FloatTensor(out_channel))
    out_r = R(in_r)
    out_r_np = out_r.data.numpy()
    y_pre = np.argmax(out_r_np,1)
    ber_nt = BER(y, y_pre, num_test)
    # print(y)
    # print(y_pre)
    return ber_nt




T = transmitter(M, n)
R = receiver(M, n)
NT = net(M, n, z_dimension)
G = generator(M, n, z_dimension)
D = discriminator(M, n)

ber_min = 1

# define loss functions
loss_nt = nn.CrossEntropyLoss()
loss_gan = nn.BCELoss()


# generate enough training data
message_idx, x, y = data_generator(M, num_train)


for epoch in range(num_epoch):
    idx = np.random.randint(num_train,size=batch_size)
    noise = np.random.randn(batch_size,z_dimension)

    # --------------------------------------------------------
    # train transmitter
    # set the invalid of g and r in net
    for name, value in NT.named_parameters():
        value.requires_grad = True
        # print(value)
        if name[0] == 'r' or name[0] == 'g':
            value.requires_grad = False
    params_t = filter(lambda p: p.requires_grad, NT.parameters())
    t_optimizer = torch.optim.Adam(params_t, lr=LR_t)
    #begin training of transmitter
    for i in range(num_transmitter):
        # idx = np.random.randint(num_train,size=batch_size)
        # noise = np.random.randn(batch_size,z_dimension)
        train_x_t = Variable(torch.Tensor(np.hstack((noise,x[idx,:]))))
        train_y_t = Variable(torch.Tensor(message_idx[idx]).long())
        out_nt = NT(train_x_t)
        t_loss = loss_nt(out_nt, train_y_t)
        # idx_net = np.argmax(out_nt.data.numpy(),1)
        # ber_net = BER(idx_net,train_y_t,batch_size)
        # print('ber of net is: ',ber_net)
        t_optimizer.zero_grad()
        t_loss.backward(retain_graph=True)
        t_optimizer.step()

    # --------------------------------------------------------
    # train GAN
    g_optimizer = torch.optim.Adam(G.parameters(), lr=LR_g)
    d_optimizer = torch.optim.Adam(D.parameters(), lr=LR_g)
    for i in range(num_GAN):

        # idx = np.random.randint(num_train,size=batch_size)
        x_input = Variable(torch.Tensor(x[idx,:]))
        encoding_result = NT.encoding(x_input).data.numpy()
        real_x = channel_AWGN(encoding_result, k, n, SNR, batch_size)

        # discriminator
        real_data = Variable(torch.cat([torch.Tensor(real_x),torch.FloatTensor(encoding_result)],1))
        real_label = Variable(torch.ones(batch_size,1))
        fake_label = Variable(torch.zeros(batch_size,1))
        # loss of real data
        real_out = D(real_data)
        d_loss_real = loss_gan(real_out,real_label)
        real_scores = d_loss_real
        # loss of fake data
        z_noise = torch.Tensor(np.random.randn(batch_size, z_dimension))
        z_coding = torch.Tensor(encoding_result)
        z = Variable(torch.cat([z_noise, z_coding],1))
        fake_data = G(z)
        fake_out = D(torch.cat([fake_data,torch.FloatTensor(encoding_result)],1))
        d_loss_fake = loss_gan(fake_out, fake_label)
        fake_scores = d_loss_fake
        # optimize
        d_loss = d_loss_fake + d_loss_real
        d_optimizer.zero_grad()
        d_loss.backward(retain_graph=True)
        d_optimizer.step()

        # generator
        z_noise = torch.Tensor(np.random.randn(batch_size, z_dimension))
        z_coding = torch.Tensor(encoding_result)
        z = Variable(torch.cat([z_noise, z_coding],1))
        fake_data_g = G(z)
        fake_out_g = D(torch.cat([fake_data_g,torch.FloatTensor(encoding_result)],1))
        g_loss = loss_gan(fake_out_g, real_label)
        # optimize
        g_optimizer.zero_grad()
        g_loss.backward(retain_graph=True)
        g_optimizer.step()

    # load the model of GAN in the origin net
    G_dict = G.state_dict()
    NT_dict = NT.state_dict()
    NT_dict.update(G_dict)
    NT.load_state_dict(NT_dict)

    # --------------------------------------------------------
    # receiver training
    # set the invalid of g and t in net
    for name, value in NT.named_parameters():
        value.requires_grad = True
        if name[0] == 't' or name[0] == 'g':
            value.requires_grad = False
    params_r = filter(lambda p: p.requires_grad, NT.parameters())
    r_optimizer = torch.optim.Adam(params_r, lr=LR_r)
    #begin training
    for i in range(num_receiver):
        # idx = np.random.randint(num_train,size=batch_size)
        # noise = np.random.randn(batch_size,z_dimension)
        train_x_r = Variable(torch.Tensor(np.hstack((noise,x[idx,:]))))
        train_y_r = Variable(torch.Tensor(message_idx[idx]).long())
        out_r = NT(train_x_r)
        r_loss = loss_nt(out_r, train_y_r)
        # idx_net = np.argmax(out_r.data.numpy(),1)
        # ber_net = BER(idx_net,train_y_r,batch_size)
        # print('ber of net is: ',ber_net)
        r_optimizer.zero_grad()
        r_loss.backward(retain_graph=True)
        r_optimizer.step()

    # --------------------------------------------------------
    # print the result
    
    # if epoch %(num_epoch//5) == 0:
    if epoch % 100 == 0:
        print('epoch: ',epoch)
        idx = np.random.randint(num_train,size=num_test)
        y_test, x_test, nouse = data_generator(M,num_test) 
        # x_test = x[idx,:]
        # y_test = message_idx[idx]

        # test the effect of generator
        x_test_1 = x_test[:,:]
        noise = np.random.randn(num_test,z_dimension)
        x_input = Variable(torch.Tensor(x_test_1))
        net_x = NT.encoding(x_input)
        noisy_x = channel_AWGN(net_x.data.numpy(),k,n,SNR,num_test)
        generator_input = torch.cat([Variable(torch.Tensor(noise)),net_x],1)
        generator_output = G(generator_input)
        # print('encoding output: ',net_x.data.numpy())
        # print('network output: ',generator_output.data.numpy())

        # if epoch % 1000 == 0:
        #     my_x = net_x.data.numpy()[0:num_plot,0]
        #     my_y = net_x.data.numpy()[0:num_plot,1]
        #     my_x_noise = generator_output.data.numpy()[0:num_plot,0]
        #     my_y_noise = generator_output.data.numpy()[0:num_plot,1]
        #     my_x_channel = noisy_x[0:num_plot,0]
        #     my_y_channel = noisy_x[0:num_plot,1]
        #     plt.figure()
        #     plt.scatter(my_x,my_y,c='b',s=10)
        #     plt.scatter(my_x_noise,my_y_noise,c='r',s=10)
        #     plt.scatter(my_x_channel,my_y_channel,c='y',s=10)
        #     plt.show()

        # print
        ber = test(M, n, k, x_test, y_test, num_test, SNR, NT, T, R)
        print('--------------------------------------------------------------------')
        print('BER: ',ber)
        print('--------------------------------------------------------------------')
        print('real: ',real_scores.data.mean(),'fake: ',fake_scores.data.mean(),'rloss: ',r_loss,'t_loss: ',t_loss)
        if ber_min > ber:
            ber_min = ber
        print('ber min is: ',ber_min)
        if ber_min < 0.3:
            break

        # # former test
        # idx = np.random.randint(num_train,size=num_test)
        # noise = np.random.randn(num_test,z_dimension)
        # train_x_t = Variable(torch.Tensor(np.hstack((x[idx,:],noise))))
        # train_y_t = Variable(torch.Tensor(message_idx[idx]).long())
        # out_nt = NT(train_x_t)
        # out_np = out_nt.data.numpy()
        # y_pre = np.argmax(out_np,1)
        # ber_nt = BER(message_idx[idx], y_pre, num_test)
        # print(message_idx[idx])
        # print(y_pre)
        # print('training error: ','receiver: ',r_loss,'transmitter: ',t_loss,'generator: ',g_loss,'discriminator: ',d_loss)
        # print('BER: ',ber_nt)

# plot the BER figure


SNR_list = np.arange(-4,21,1)
leng = SNR_list.shape[0]
ber_result = np.zeros([leng])
for i in range(leng):
    SNR = 10**(SNR_list[i]/10)
    ber_result[i] = test(M, n, k, x_test, y_test, num_test, SNR, NT, T, R)

ber_refer = [0.4958,0.42262,0.3438,0.26389,0.18801,0.12099,0.06883,0.03354,0.0134,0.00436,0.00108,0.00018,1e-05,0,0,0,0,0,0,0,0,0,0,0,0]
plt.semilogy(SNR_list,ber_result,'ro-.',linewidth=2)
plt.semilogy(SNR_list,ber_refer,'bv:',linewidth=2)
plt.axis([0,20,0.001,1])
label = ["the paper of LiYe about AWGN channel","7 layer autoencoder"] 
plt.legend(label, loc = 0, ncol = 1)
plt.grid(True) ##增加格点
plt.xlabel('SNR (dB)')
plt.ylabel('BLER')
plt.show()

torch.save(NT.state_dict(),'./BER/SNR-0-0.3')
sio.savemat('GAN_0_0.03_%d%d'%(n,k),{'SNR_dB':SNR_list, 'ber':ber_result})



    






