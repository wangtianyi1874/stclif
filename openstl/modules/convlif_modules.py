from spikingjelly.activation_based import surrogate
from typing import Callable
import torch
import torch.nn as nn
from spikingjelly.activation_based import layer,neuron
import math
import torch
import torch.nn as nn





class SpikeAct_extended(torch.autograd.Function):
    '''
    solving the non-differentiable term of the Heavisde function
    '''
    @staticmethod
    def forward(ctx, input):
        ctx.save_for_backward(input)
        # if input = u > Vth then output = 1
        output = torch.gt(input, 0.)
        return output.float()

    @staticmethod
    def backward(ctx, grad_output):
        input = ctx.saved_tensors
        grad_input = grad_output.clone() 
        # input.shape
        input=torch.stack(input)
        hu = abs(input) < 0.5
        hu = hu.float()
        return grad_input * hu

class ArchAct(torch.autograd.Function):
    @staticmethod
    def forward(ctx, input):
        ctx.save_for_backward(input)
        output = torch.gt(input, 0.5)
        return output.float()
    @staticmethod
    def backward(ctx, grad_output):
        input = ctx.saved_tensors
        grad_input = grad_output.clone()
        return grad_input


class LIFSpike_CW(nn.Module):
    '''
    gated spiking neuron
    '''
    def __init__(self, inplace, **kwargs):
        super(LIFSpike_CW, self).__init__()
        self.T = kwargs['t']
        self.soft_mode = kwargs['soft_mode']
        self.static_gate = kwargs['static_gate']
        self.static_param = kwargs['static_param']
        self.time_wise = kwargs['time_wise']
        self.plane = inplace
        #c
        self.alpha, self.beta, self.gamma = [nn.Parameter(- math.log(1 / ((i - 0.5)*0.5+0.5) - 1) * torch.ones(self.plane, dtype=torch.float))
                                                 for i in kwargs['gate']]

        self.tau, self.Vth, self.leak = [nn.Parameter(- math.log(1 / i - 1) * torch.ones(self.plane, dtype=torch.float))
                              for i in kwargs['param'][:-1]]
        self.reVth = nn.Parameter(- math.log(1 / kwargs['param'][1] - 1) * torch.ones(self.plane, dtype=torch.float))
        #t, c
        self.conduct = [nn.Parameter(- math.log(1 / i - 1) * torch.ones((self.T, self.plane), dtype=torch.float))
                                   for i in kwargs['param'][3:]][0]

    def forward(self, x_t,h_t,g_t): #t, b, c, h, w

        # u = torch.zeros(x.shape[1:], device=x.device)
        # out = torch.zeros(x.shape, device=x.device)
        # for step in range(self.T):
        h_t, out = self.extended_state_update(h_t, g_t, x_t,
                                                      tau=self.tau.sigmoid(),
                                                      Vth=self.Vth.sigmoid(),
                                                      leak=self.leak.sigmoid(),
                                                      conduct=self.conduct[0].sigmoid(),
                                                      reVth=self.reVth.sigmoid())
        return out,h_t

    #[b, c, h, w]  * [c]
    def extended_state_update(self, u_t_n1, o_t_n1, W_mul_o_t_n1, tau, Vth, leak, conduct, reVth):
        # print(W_mul_o_t_n1.shape, self.alpha[None, :, None, None].sigmoid().shape)
        if self.static_gate:
            if self.soft_mode:
                al, be, ga = self.alpha.view(1, -1, 1, 1).clone().detach().sigmoid(), self.beta.view(1, -1, 1, 1).clone().detach().sigmoid(), self.gamma.view(1, -1, 1, 1).clone().detach().sigmoid()
            else:
                al, be, ga = self.alpha.view(1, -1, 1, 1).clone().detach().gt(0.).float(), self.beta.view(1, -1, 1, 1).clone().detach().gt(0.).float(), self.gamma.view(1, -1, 1, 1).clone().detach().gt(0.).float()
        else:
            if self.soft_mode:
                al, be, ga = self.alpha.view(1, -1, 1, 1).sigmoid(), self.beta.view(1, -1, 1, 1).sigmoid(), self.gamma.view(1, -1, 1, 1).sigmoid()
            else:
                al, be, ga = ArchAct.apply(self.alpha.view(1, -1, 1, 1).sigmoid()), ArchAct.apply(self.beta.view(1, -1, 1, 1).sigmoid()), ArchAct.apply(self.gamma.view(1, -1, 1, 1).sigmoid())

        # I_t1 = W_mul_o_t_n1 + be * I_t0 * self.conduct.sigmoid()#原先
        I_t1 = W_mul_o_t_n1 * (1 - be * (1 - conduct[None, :, None, None]))
        u_t1_n1 = ((1 - al * (1 - tau[None, :, None, None])) * u_t_n1 * (1 - ga * o_t_n1.clone()) - (1 - al) * leak[None, :, None, None]) + \
                  I_t1 - (1 - ga) * reVth[None, :, None, None] * o_t_n1.clone()
        o_t1_n1 = SpikeAct_extended.apply(u_t1_n1 - Vth[None, :, None, None])
        return u_t1_n1, o_t1_n1

    def _initialize_params(self, **kwargs):
        self.mid_gate_mode = True
        self.tau.copy_(torch.tensor(- math.log(1 / kwargs['param'][0] - 1), dtype=torch.float, device=self.tau.device))
        self.Vth.copy_(torch.tensor(- math.log(1 / kwargs['param'][1] - 1), dtype=torch.float, device=self.Vth.device))
        self.reVth.copy_(torch.tensor(- math.log(1 / kwargs['param'][1] - 1), dtype=torch.float, device=self.reVth.device))

        self.leak.copy_(- math.log(1 / kwargs['param'][2] - 1) * torch.ones(self.T, dtype=torch.float, device=self.leak.device))
        self.conduct.copy_(- math.log(1 / kwargs['param'][3] - 1) * torch.ones(self.T, dtype=torch.float, device=self.conduct.device))


# glif
class GLIFCell(nn.Module):
    r"""GLIF

    Implementation of `Glif: A unified gated leaky integrate-and-fire neuron for spiking neural
    networks.`

    """

    def __init__(self, in_channel, num_hidden, height, width, filter_size, stride, layer_norm,t_m: float = 2., v_threshold: float = 1., v_reset: float = 0.,
                 surrogate_function: Callable = surrogate.Sigmoid()):
        super(GLIFCell, self).__init__()

        steps = 4
        Vth = 0.5  #  V_threshold
        tau = 0.25  # exponential decay coefficient
        conduct = 0.5 # time-dependent synaptic weight
        linear_decay = Vth/(steps * 2)  #linear decay coefficient
        self.num_hidden = num_hidden
        self.padding = filter_size // 2

        initial_dict = {'gate': [0.6, 0.8, 0.6], 'param': [tau, Vth, linear_decay, conduct],
               't': 1, 'static_gate': True, 'static_param': False, 'time_wise': False, 'soft_mode': True}
        self.lif_param = initial_dict
        groups=16

        # Both self.conv1 and self.downsample layers downsample the input when stride != 1
        self.conv_x = nn.Sequential(
        nn.Conv2d(in_channel, num_hidden, kernel_size=filter_size,
                    stride=stride, padding=self.padding, bias=False),
        # nn.Conv2d(in_channel, num_hidden, kernel_size=filter_size,
        #             stride=stride, padding=self.padding, bias=False,groups=in_channel),
        # tdBatchNorm(num_hidden),
        nn.GroupNorm(num_groups=groups, num_channels=num_hidden),
        # LIFSpike_CW(num_hidden, **self.lif_param)
        )        
        self.glif=LIFSpike_CW(num_hidden, **self.lif_param)

        
    def forward(self, x_t, h_t,g_t):
        # print(x.shape)
        out = self.conv_x(x_t)
        out,h_t=self.glif(out,h_t,g_t)
        # print(out.shape)
        return out,h_t



        
class LIFCell(nn.Module):
    r"""
    LIF model

    """
    def __init__(self, in_channel, num_hidden, height, width, filter_size, stride, layer_norm,t_m: float = 2., v_threshold: float = 1., v_reset: float = 0.,
                 surrogate_function: Callable = surrogate.Sigmoid()):
        super(LIFCell, self).__init__()

        self.num_hidden = num_hidden
        self.padding = filter_size // 2
        self._forget_bias = 1.0
        
        self.v_threshold=v_threshold
        self.v_reset=v_reset
        self.fire=surrogate_function

        self.t_m=1/t_m

        groups=16

        self.conv_x = nn.Sequential(
            nn.Conv2d(in_channel, num_hidden, kernel_size=filter_size,
                        stride=stride, padding=self.padding, bias=False),
            nn.GroupNorm(num_groups=groups, num_channels=num_hidden)
        )          
                
    def forward(self, x_t, h_t,g_t):
        h_s = h_t*self.t_m+self.conv_x(x_t)*(1-self.t_m)
        s_t = self.fire(h_s-self.v_threshold)
        h_t = h_s-s_t
        return s_t,h_t
    
class PLIFCell(nn.Module):
    r"""PLIF model

    Implementation of `Incorporating learnable membrane time constant
    to enhance learning of spiking neural networks.`_.

    """
    def __init__(self, in_channel, num_hidden, height, width, filter_size, stride, layer_norm,t_m: float = 2., v_threshold: float = 1., v_reset: float = 0.,
                 surrogate_function: Callable = surrogate.Sigmoid()):
        super(PLIFCell, self).__init__()

        self.num_hidden = num_hidden
        self.padding = filter_size // 2
        self._forget_bias = 1.0
        
        self.v_threshold=v_threshold
        self.v_reset=v_reset
        self.fire=surrogate_function

        self.t_m=nn.Parameter(torch.zeros(1))

        groups=16

        self.conv_x = nn.Sequential(
            nn.Conv2d(in_channel, num_hidden, kernel_size=filter_size,
                        stride=stride, padding=self.padding, bias=False),
            nn.GroupNorm(num_groups=groups, num_channels=num_hidden)
        )          
                
    def forward(self, x_t, h_t,g_t):

        h_s = h_t*self.t_m.sigmoid()+self.conv_x(x_t)*(1-self.t_m)
        s_t = self.fire(h_s-self.v_threshold)
        h_t = h_s-s_t
        return s_t,h_t

class TCLIFCell(nn.Module):
    r"""TCLIF model

    Implementation of `Tc-lif: A two-compartment spiking neuron model
    for long-term sequential modelling.`

    """
    def __init__(self, in_channel, num_hidden, height, width, filter_size, stride, layer_norm,t_m: float = 1., v_threshold: float = 1., v_reset: float = 0.,
                 surrogate_function: Callable = surrogate.Sigmoid()):
        super(TCLIFCell, self).__init__()

        self.num_hidden = num_hidden
        self.padding = filter_size // 2
        self._forget_bias = 1.0

        self.v_threshold=v_threshold
        self.v_reset=v_reset
        self.fire=surrogate_function
        groups=16

        self.t_m=nn.Parameter(torch.zeros(1))
        
        self.s_u=nn.Parameter(torch.zeros(1))
        self.s_p=nn.Parameter(torch.zeros(1))

        self.d_u=nn.Parameter(torch.zeros(1))
        self.d_p=nn.Parameter(torch.zeros(1))

        self.conv_x = nn.Sequential(
            nn.Conv2d(in_channel, num_hidden, kernel_size=filter_size,
                        stride=stride, padding=self.padding, bias=False),
            nn.GroupNorm(num_groups=groups, num_channels=num_hidden)
        )

    def forward(self, x_t, h_t,d_t,g_t):
        x_s = self.conv_x(x_t)
        d_t = d_t-h_t*self.s_u.sigmoid()+x_s-self.t_m*g_t.detach()
        h_t = h_t+d_t*self.s_p.sigmoid()-g_t.detach()
        s_t = self.fire(h_t-self.v_threshold)
        return s_t,h_t,d_t
    
class LMHCell(nn.Module):
    r"""LM-H model

    Implementation of `A Progressive Training Framework for Spiking Neural Networks 
    with Learnable Multi-hierarchical Model <https://openreview.net/forum?id=g52tgL8jy6>`_.

    """
    def __init__(self, in_channel, num_hidden, height, width, filter_size, stride, layer_norm,t_m: float = 1., v_threshold: float = 1., v_reset: float = 0.,
                 surrogate_function: Callable = surrogate.Sigmoid()):
        super(LMHCell, self).__init__()

        self.num_hidden = num_hidden
        self.padding = filter_size // 2
        self._forget_bias = 1.0

        self.v_threshold=v_threshold
        self.v_reset=v_reset
        self.fire=surrogate_function
        groups=16

        self.t_m=nn.Parameter(torch.zeros(1))
        
        self.s_u=nn.Parameter(torch.zeros(1))
        self.s_p=nn.Parameter(torch.zeros(1))

        self.d_u=nn.Parameter(torch.zeros(1))
        self.d_p=nn.Parameter(torch.zeros(1))

        self.conv_x = nn.Sequential(
            nn.Conv2d(in_channel, num_hidden, kernel_size=filter_size,
                        stride=stride, padding=self.padding, bias=False),
            nn.GroupNorm(num_groups=groups, num_channels=num_hidden)
        )
 
    def forward(self, x_t, h_t,d_t,g_t):
        x_s = self.conv_x(x_t)
        d_t = d_t*(self.d_u.sigmoid()-0.5)+h_t*(self.s_u.sigmoid()-0.5)+x_s
        h_s = d_t*(self.d_p.sigmoid()+0.5)+h_t*(self.s_p.sigmoid()+0.5)
        s_t = self.fire(h_s-self.v_threshold)
        h_t = h_s-s_t.detach()
        return s_t,h_t,d_t

class STCLIFCell(nn.Module):
    r"""STC-LIF model

    Implementation of "Autaptic Synaptic Circuit Enhances Spatio-temporal Predictive
    Learning of Spiking Neural Networks."

    """
    def __init__(self, in_channel, num_hidden, height, width, filter_size, stride, layer_norm,gradient_detach = 1,t_m: float = 2., v_threshold: float = 1., v_reset: float = 0.,
                 surrogate_function: Callable = surrogate.Sigmoid()):
        super(STCLIFCell, self).__init__()

        self.num_hidden = num_hidden
        self.padding = filter_size // 2

        
        self.gradient_detach=gradient_detach
        self.v_threshold=v_threshold
        self.v_reset=v_reset
        self.fire=surrogate_function
        groups=16

        self.conv_x = nn.Sequential(
            nn.Conv2d(in_channel, num_hidden, kernel_size=filter_size,
                        stride=stride, padding=self.padding, bias=False),
            nn.GroupNorm(num_groups=groups, num_channels=num_hidden)
        )
 
        self.conv_h = nn.Sequential(
            nn.Conv2d(num_hidden, num_hidden, kernel_size=filter_size,
            stride=stride, padding=self.padding, bias=False,groups=groups),
            nn.GroupNorm(num_groups=groups, num_channels=num_hidden)
        )
        self.conv_v = nn.Sequential(
            nn.Conv2d(num_hidden, num_hidden, kernel_size=filter_size,
            stride=stride, padding=self.padding, bias=False,groups=groups),
            nn.GroupNorm(num_groups=groups, num_channels=num_hidden)        
        )            
                
    def forward(self, x_t, h_t,g_t):
        if self.gradient_detach:
            h_t = h_t*(1+torch.tanh(self.conv_h(g_t.detach())))+self.conv_x(x_t)*(1+torch.tanh(self.conv_v(g_t.detach())))
        else:
            h_t = h_t*(1+torch.tanh(self.conv_h(g_t)))+self.conv_x(x_t)*(1+torch.tanh(self.conv_v(g_t)))
        s_t = self.fire(h_t-self.v_threshold)
        h_t = h_t-s_t

        return s_t,h_t
    
class STCPLIFCell(nn.Module):
    r"""STC-PLIF model

    Implementation of "Autaptic Synaptic Circuit Enhances Spatio-temporal Predictive
    Learning of Spiking Neural Networks."

    """

    def __init__(self, in_channel, num_hidden, height, width, filter_size, stride, layer_norm,gradient_detach=1,t_m: float = 2., v_threshold: float = 1., v_reset: float = 0.,
                 surrogate_function: Callable = surrogate.Sigmoid()):
        super(STCPLIFCell, self).__init__()

        self.num_hidden = num_hidden
        self.padding = filter_size // 2
        self.gradient_detach=gradient_detach
        
        self.v_threshold=v_threshold
        self.v_reset=v_reset
        self.fire=surrogate_function
        self.t_m=nn.Parameter(torch.zeros(1))
        groups=16

        self.conv_x = nn.Sequential(
            nn.Conv2d(in_channel, num_hidden, kernel_size=filter_size,
                        stride=stride, padding=self.padding, bias=False),
            nn.GroupNorm(num_groups=groups, num_channels=num_hidden)
        )
 
        self.conv_h = nn.Sequential(
            nn.Conv2d(num_hidden, num_hidden, kernel_size=filter_size,
            stride=stride, padding=self.padding, bias=False,groups=groups),
            nn.GroupNorm(num_groups=groups, num_channels=num_hidden)
        )
        self.conv_v = nn.Sequential(
            nn.Conv2d(num_hidden, num_hidden, kernel_size=filter_size,
            stride=stride, padding=self.padding, bias=False,groups=groups),
            nn.GroupNorm(num_groups=groups, num_channels=num_hidden)        
        )            
                
    def forward(self, x_t, h_t,g_t):
        if self.gradient_detach:
            h_t = h_t*(1+torch.tanh(self.conv_h(g_t.detach())))*self.t_m.sigmoid()+self.conv_x(x_t)*(1+torch.tanh(self.conv_v(g_t.detach())))*(1-self.t_m.sigmoid())
        else:
            h_t = h_t*(1+torch.tanh(self.conv_h(g_t)))*self.t_m.sigmoid()+self.conv_x(x_t)*(1+torch.tanh(self.conv_v(g_t)))*(1-self.t_m.sigmoid())
        s_t = self.fire(h_t-self.v_threshold)
        h_t = h_t-s_t

        return s_t,h_t
    
class STCLMHCell(nn.Module):

    r"""STC-LMH model

    Implementation of "Autaptic Synaptic Circuit Enhances Spatio-temporal Predictive
    Learning of Spiking Neural Networks."

    """

    def __init__(self, in_channel, num_hidden, height, width, filter_size, stride, layer_norm,gradient_detach=1,t_m: float = 2., v_threshold: float = 1., v_reset: float = 0.,
                 surrogate_function: Callable = surrogate.Sigmoid()):
        super(STCLMHCell, self).__init__()

        self.num_hidden = num_hidden
        self.padding = filter_size // 2
        self.gradient_detach=gradient_detach
        
        self.v_threshold=v_threshold
        self.v_reset=v_reset
        self.fire=surrogate_function
        self.t_m=nn.Parameter(torch.zeros(1))
        groups=16

        self.t_m=nn.Parameter(torch.zeros(1))
        
        self.s_u=nn.Parameter(torch.zeros(1))
        self.s_p=nn.Parameter(torch.zeros(1))

        self.d_u=nn.Parameter(torch.zeros(1))
        self.d_p=nn.Parameter(torch.zeros(1))

        self.conv_x = nn.Sequential(
            nn.Conv2d(in_channel, num_hidden, kernel_size=filter_size,
                        stride=stride, padding=self.padding, bias=False),
            nn.GroupNorm(num_groups=groups, num_channels=num_hidden)
        )
 
        self.conv_h = nn.Sequential(
            nn.Conv2d(num_hidden, num_hidden, kernel_size=filter_size,
            stride=stride, padding=self.padding, bias=False,groups=groups),
            nn.GroupNorm(num_groups=groups, num_channels=num_hidden)
        )
        self.conv_v = nn.Sequential(
            nn.Conv2d(num_hidden, num_hidden, kernel_size=filter_size,
            stride=stride, padding=self.padding, bias=False,groups=groups),
            nn.GroupNorm(num_groups=groups, num_channels=num_hidden)        
        )            

    def forward(self, x_t, h_t,d_t,g_t):
        if self.gradient_detach:
            x_g = torch.tanh(self.conv_h(g_t.detach()))
            x_v = torch.tanh(self.conv_v(g_t.detach()))
        else:
            x_g = torch.tanh(self.conv_h(g_t))
            x_v = torch.tanh(self.conv_v(g_t))
            
        x_s = self.conv_x(x_t)*(1+x_v)

        d_t = d_t*(self.d_u.sigmoid()-0.5)+h_t*(self.s_u.sigmoid()-0.5)+x_s
        h_t=h_t*(1+x_g)
        h_s = d_t*(self.d_p.sigmoid()+0.5)+h_t*(self.s_p.sigmoid()+0.5)
        s_t = self.fire(h_s-self.v_threshold)

        h_t = h_s-s_t.detach()
        
        return s_t,h_t,d_t
