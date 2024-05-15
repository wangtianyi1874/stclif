import torch
import torch.nn as nn

from openstl.modules  import LIFCell,PLIFCell,LMHCell,TCLIFCell,STCLIFCell,STCPLIFCell,STCLMHCell,GLIFCell

class GLIF_Model(nn.Module):


    def __init__(self, num_layers, num_hidden, configs, **kwargs):
        super(GLIF_Model, self).__init__()
        T, C, H, W = configs.in_shape

        self.configs = configs
        self.frame_channel = configs.patch_size * configs.patch_size * C         #帧由1x64x64打乱为16x16x16
        self.num_layers = num_layers
        self.num_hidden = num_hidden
        cell_list = []

        # print('okk')
        # print('num_hidden')
        height = H // configs.patch_size
        width = W // configs.patch_size
        self.MSE_criterion = nn.MSELoss()

        for i in range(num_layers):
            in_channel = self.frame_channel if i == 0 else num_hidden[i - 1]
            cell_list.append(
                GLIFCell(in_channel, num_hidden[i], height, height, configs.filter_size,
                                       configs.stride, configs.layer_norm)
            )
        self.cell_list = nn.ModuleList(cell_list)
        self.conv_last = nn.Conv2d(num_hidden[num_layers - 1], self.frame_channel,
                                   kernel_size=1, stride=1, padding=0, bias=False)

    def forward(self, frames_tensor, mask_true, **kwargs):
        # [batch, length, height, width, channel] -> [batch, length, channel, height, width]
        frames = frames_tensor.permute(0, 1, 4, 2, 3).contiguous()
        mask_true = mask_true.permute(0, 1, 4, 2, 3).contiguous()

        batch = frames.shape[0]
        height = frames.shape[3]
        width = frames.shape[4]

        next_frames = []
        h_t = []
        s_t = []
        d_t = []

        for i in range(self.num_layers):
            zeros = torch.zeros([batch, self.num_hidden[i], height, height]).to(self.configs.device)
            h_t.append(zeros)
            s_t.append(zeros)
            # d_t.append(zeros)

        for t in range(self.configs.pre_seq_length + self.configs.aft_seq_length - 1):
            # reverse schedule sampling
       
            if self.configs.reverse_scheduled_sampling == 1:
                if t == 0:
                    net = frames[:, t]
                else:
                    net = mask_true[:, t - 1] * frames[:, t] + (1 - mask_true[:, t - 1]) * x_gen
            else:
                if t < self.configs.pre_seq_length:
                    net = frames[:, t]
                else:
                    net = mask_true[:, t - self.configs.pre_seq_length] * frames[:, t] + \
                            (1 - mask_true[:, t - self.configs.pre_seq_length]) * x_gen

            s_t[0], h_t[0] = self.cell_list[0](net, h_t[0],s_t[0])

            for i in range(1, self.num_layers):
                s_t[i], h_t[i] = self.cell_list[i](s_t[i-1], h_t[i],s_t[i])


            x_gen = self.conv_last(s_t[self.num_layers - 1])
            next_frames.append(x_gen)

        # [length, batch, channel, height, width] -> [batch, length, height, width, channel]
        next_frames = torch.stack(next_frames, dim=0).permute(1, 0, 3, 4, 2).contiguous()
        if kwargs.get('return_loss', True):
            loss = self.MSE_criterion(next_frames, frames_tensor[:, 1:])
        else:
            loss = None

        return next_frames, loss


class LIF_Model(nn.Module):
 

    def __init__(self, num_layers, num_hidden, configs, **kwargs):
        super(LIF_Model, self).__init__()
        T, C, H, W = configs.in_shape

        self.configs = configs
        self.frame_channel = configs.patch_size * configs.patch_size * C         #帧由1x64x64打乱为16x16x16
        self.num_layers = num_layers
        self.num_hidden = num_hidden
        cell_list = []

        # print('okk')
        # print('num_hidden')
        height = H // configs.patch_size
        width = W // configs.patch_size
        self.MSE_criterion = nn.MSELoss()

        for i in range(num_layers):
            in_channel = self.frame_channel if i == 0 else num_hidden[i - 1]
            cell_list.append(
                LIFCell(in_channel, num_hidden[i], height, height, configs.filter_size,
                                       configs.stride, configs.layer_norm)
            )
        self.cell_list = nn.ModuleList(cell_list)
        self.conv_last = nn.Conv2d(num_hidden[num_layers - 1], self.frame_channel,
                                   kernel_size=1, stride=1, padding=0, bias=False)

    def forward(self, frames_tensor, mask_true, **kwargs):
        # [batch, length, height, width, channel] -> [batch, length, channel, height, width]
        frames = frames_tensor.permute(0, 1, 4, 2, 3).contiguous()
        mask_true = mask_true.permute(0, 1, 4, 2, 3).contiguous()

        batch = frames.shape[0]
        height = frames.shape[3]
        width = frames.shape[4]

        next_frames = []
        h_t = []
        s_t = []
        d_t = []

        for i in range(self.num_layers):
            zeros = torch.zeros([batch, self.num_hidden[i], height, height]).to(self.configs.device)
            h_t.append(zeros)
            s_t.append(zeros)
            # d_t.append(zeros)

        for t in range(self.configs.pre_seq_length + self.configs.aft_seq_length - 1):
            # reverse schedule sampling
       
            if self.configs.reverse_scheduled_sampling == 1:
                if t == 0:
                    net = frames[:, t]
                else:
                    net = mask_true[:, t - 1] * frames[:, t] + (1 - mask_true[:, t - 1]) * x_gen
            else:
                if t < self.configs.pre_seq_length:
                    net = frames[:, t]
                else:
                    net = mask_true[:, t - self.configs.pre_seq_length] * frames[:, t] + \
                            (1 - mask_true[:, t - self.configs.pre_seq_length]) * x_gen

            s_t[0], h_t[0] = self.cell_list[0](net, h_t[0],s_t[0])

            for i in range(1, self.num_layers):
                s_t[i], h_t[i] = self.cell_list[i](s_t[i-1], h_t[i],s_t[i])


            x_gen = self.conv_last(s_t[self.num_layers - 1])
            next_frames.append(x_gen)

        # [length, batch, channel, height, width] -> [batch, length, height, width, channel]
        next_frames = torch.stack(next_frames, dim=0).permute(1, 0, 3, 4, 2).contiguous()
        if kwargs.get('return_loss', True):
            loss = self.MSE_criterion(next_frames, frames_tensor[:, 1:])
        else:
            loss = None

        return next_frames, loss

class PLIF_Model(nn.Module):
 

    def __init__(self, num_layers, num_hidden, configs, **kwargs):
        super(PLIF_Model, self).__init__()
        T, C, H, W = configs.in_shape

        self.configs = configs
        self.frame_channel = configs.patch_size * configs.patch_size * C         #帧由1x64x64打乱为16x16x16
        self.num_layers = num_layers
        self.num_hidden = num_hidden
        cell_list = []

        # print('okk')
        # print('num_hidden')
        height = H // configs.patch_size
        width = W // configs.patch_size
        self.MSE_criterion = nn.MSELoss()

        for i in range(num_layers):
            in_channel = self.frame_channel if i == 0 else num_hidden[i - 1]
            cell_list.append(
                PLIFCell(in_channel, num_hidden[i], height, height, configs.filter_size,
                                       configs.stride, configs.layer_norm)
            )
        self.cell_list = nn.ModuleList(cell_list)
        self.conv_last = nn.Conv2d(num_hidden[num_layers - 1], self.frame_channel,
                                   kernel_size=1, stride=1, padding=0, bias=False)

    def forward(self, frames_tensor, mask_true, **kwargs):
        # [batch, length, height, width, channel] -> [batch, length, channel, height, width]
        frames = frames_tensor.permute(0, 1, 4, 2, 3).contiguous()
        mask_true = mask_true.permute(0, 1, 4, 2, 3).contiguous()

        batch = frames.shape[0]
        height = frames.shape[3]
        width = frames.shape[4]

        next_frames = []
        h_t = []
        s_t = []
        d_t = []

        for i in range(self.num_layers):
            zeros = torch.zeros([batch, self.num_hidden[i], height, height]).to(self.configs.device)
            h_t.append(zeros)
            s_t.append(zeros)
            # d_t.append(zeros)

        for t in range(self.configs.pre_seq_length + self.configs.aft_seq_length - 1):
            # reverse schedule sampling
       
            if self.configs.reverse_scheduled_sampling == 1:
                if t == 0:
                    net = frames[:, t]
                else:
                    net = mask_true[:, t - 1] * frames[:, t] + (1 - mask_true[:, t - 1]) * x_gen
            else:
                if t < self.configs.pre_seq_length:
                    net = frames[:, t]
                else:
                    net = mask_true[:, t - self.configs.pre_seq_length] * frames[:, t] + \
                            (1 - mask_true[:, t - self.configs.pre_seq_length]) * x_gen

            s_t[0], h_t[0] = self.cell_list[0](net, h_t[0],s_t[0])

            for i in range(1, self.num_layers):
                s_t[i], h_t[i] = self.cell_list[i](s_t[i-1], h_t[i],s_t[i])


            x_gen = self.conv_last(s_t[self.num_layers - 1])
            next_frames.append(x_gen)

        # [length, batch, channel, height, width] -> [batch, length, height, width, channel]
        next_frames = torch.stack(next_frames, dim=0).permute(1, 0, 3, 4, 2).contiguous()
        if kwargs.get('return_loss', True):
            loss = self.MSE_criterion(next_frames, frames_tensor[:, 1:])
        else:
            loss = None

        return next_frames, loss
    
class STCLIF_Model(nn.Module):

    def __init__(self, num_layers, num_hidden, configs, **kwargs):
        super(STCLIF_Model, self).__init__()
        T, C, H, W = configs.in_shape

        self.configs = configs
        self.frame_channel = configs.patch_size * configs.patch_size * C         #帧由1x64x64打乱为16x16x16
        self.num_layers = num_layers
        self.num_hidden = num_hidden
        cell_list = []

        # print('okk')
        # print('num_hidden')
        height = H // configs.patch_size
        width = W // configs.patch_size
        self.MSE_criterion = nn.MSELoss()

        for i in range(num_layers):
            in_channel = self.frame_channel if i == 0 else num_hidden[i - 1]
            cell_list.append(
                STCLIFCell(in_channel, num_hidden[i], height, height, configs.filter_size,
                                       configs.stride, configs.layer_norm,configs.gradient_detach)
            )
        self.cell_list = nn.ModuleList(cell_list)
        self.conv_last = nn.Conv2d(num_hidden[num_layers - 1], self.frame_channel,
                                   kernel_size=1, stride=1, padding=0, bias=False)

    def forward(self, frames_tensor, mask_true, **kwargs):
        # [batch, length, height, width, channel] -> [batch, length, channel, height, width]
        frames = frames_tensor.permute(0, 1, 4, 2, 3).contiguous()
        mask_true = mask_true.permute(0, 1, 4, 2, 3).contiguous()

        batch = frames.shape[0]
        height = frames.shape[3]
        width = frames.shape[4]

        next_frames = []
        h_t = []
        s_t = []
        d_t = []

        for i in range(self.num_layers):
            zeros = torch.zeros([batch, self.num_hidden[i], height, height]).to(self.configs.device)
            h_t.append(zeros)
            s_t.append(zeros)
            # d_t.append(zeros)

        for t in range(self.configs.pre_seq_length + self.configs.aft_seq_length - 1):
            # reverse schedule sampling
       
            if self.configs.reverse_scheduled_sampling == 1:
                if t == 0:
                    net = frames[:, t]
                else:
                    net = mask_true[:, t - 1] * frames[:, t] + (1 - mask_true[:, t - 1]) * x_gen
            else:
                if t < self.configs.pre_seq_length:
                    net = frames[:, t]
                else:
                    net = mask_true[:, t - self.configs.pre_seq_length] * frames[:, t] + \
                            (1 - mask_true[:, t - self.configs.pre_seq_length]) * x_gen

            s_t[0], h_t[0] = self.cell_list[0](net, h_t[0],s_t[0])

            for i in range(1, self.num_layers):
                s_t[i], h_t[i] = self.cell_list[i](s_t[i-1], h_t[i],s_t[i])


            x_gen = self.conv_last(s_t[self.num_layers - 1])
            next_frames.append(x_gen)

        # [length, batch, channel, height, width] -> [batch, length, height, width, channel]
        next_frames = torch.stack(next_frames, dim=0).permute(1, 0, 3, 4, 2).contiguous()
        if kwargs.get('return_loss', True):
            loss = self.MSE_criterion(next_frames, frames_tensor[:, 1:])
        else:
            loss = None

        return next_frames, loss

class STCPLIF_Model(nn.Module):
 

    def __init__(self, num_layers, num_hidden, configs, **kwargs):
        super(STCPLIF_Model, self).__init__()
        T, C, H, W = configs.in_shape

        self.configs = configs
        self.frame_channel = configs.patch_size * configs.patch_size * C         #帧由1x64x64打乱为16x16x16
        self.num_layers = num_layers
        self.num_hidden = num_hidden
        cell_list = []

        # print('okk')
        # print('num_hidden')
        height = H // configs.patch_size
        width = W // configs.patch_size
        self.MSE_criterion = nn.MSELoss()

        for i in range(num_layers):
            in_channel = self.frame_channel if i == 0 else num_hidden[i - 1]
            cell_list.append(
                STCPLIFCell(in_channel, num_hidden[i], height, height, configs.filter_size,
                                       configs.stride, configs.layer_norm,configs.gradient_detach)
            )
        self.cell_list = nn.ModuleList(cell_list)
        self.conv_last = nn.Conv2d(num_hidden[num_layers - 1], self.frame_channel,
                                   kernel_size=1, stride=1, padding=0, bias=False)

    def forward(self, frames_tensor, mask_true, **kwargs):
        # [batch, length, height, width, channel] -> [batch, length, channel, height, width]
        frames = frames_tensor.permute(0, 1, 4, 2, 3).contiguous()
        mask_true = mask_true.permute(0, 1, 4, 2, 3).contiguous()

        batch = frames.shape[0]
        height = frames.shape[3]
        width = frames.shape[4]

        next_frames = []
        h_t = []
        s_t = []
        d_t = []

        for i in range(self.num_layers):
            zeros = torch.zeros([batch, self.num_hidden[i], height, height]).to(self.configs.device)
            h_t.append(zeros)
            s_t.append(zeros)
            # d_t.append(zeros)

        for t in range(self.configs.pre_seq_length + self.configs.aft_seq_length - 1):
            # reverse schedule sampling
       
            if self.configs.reverse_scheduled_sampling == 1:
                if t == 0:
                    net = frames[:, t]
                else:
                    net = mask_true[:, t - 1] * frames[:, t] + (1 - mask_true[:, t - 1]) * x_gen
            else:
                if t < self.configs.pre_seq_length:
                    net = frames[:, t]
                else:
                    net = mask_true[:, t - self.configs.pre_seq_length] * frames[:, t] + \
                            (1 - mask_true[:, t - self.configs.pre_seq_length]) * x_gen

            s_t[0], h_t[0] = self.cell_list[0](net, h_t[0],s_t[0])

            for i in range(1, self.num_layers):
                s_t[i], h_t[i] = self.cell_list[i](s_t[i-1], h_t[i],s_t[i])


            x_gen = self.conv_last(s_t[self.num_layers - 1])
            next_frames.append(x_gen)

        # [length, batch, channel, height, width] -> [batch, length, height, width, channel]
        next_frames = torch.stack(next_frames, dim=0).permute(1, 0, 3, 4, 2).contiguous()
        if kwargs.get('return_loss', True):
            loss = self.MSE_criterion(next_frames, frames_tensor[:, 1:])
        else:
            loss = None

        return next_frames, loss

class TCLIF_Model(nn.Module):
  

    def __init__(self, num_layers, num_hidden, configs, **kwargs):
        super(TCLIF_Model, self).__init__()
        T, C, H, W = configs.in_shape

        self.configs = configs
        self.frame_channel = configs.patch_size * configs.patch_size * C         #帧由1x64x64打乱为16x16x16
        self.num_layers = num_layers
        self.num_hidden = num_hidden
        cell_list = []

        # print('okk')
        # print('num_hidden')
        height = H // configs.patch_size
        width = W // configs.patch_size
        self.MSE_criterion = nn.MSELoss()

        for i in range(num_layers):
            in_channel = self.frame_channel if i == 0 else num_hidden[i - 1]
            cell_list.append(
                TCLIFCell(in_channel, num_hidden[i], height, height, configs.filter_size,
                                       configs.stride, configs.layer_norm)
            )
        self.cell_list = nn.ModuleList(cell_list)
        self.conv_last = nn.Conv2d(num_hidden[num_layers - 1], self.frame_channel,
                                   kernel_size=1, stride=1, padding=0, bias=False)

    def forward(self, frames_tensor, mask_true, **kwargs):
        # [batch, length, height, width, channel] -> [batch, length, channel, height, width]
        frames = frames_tensor.permute(0, 1, 4, 2, 3).contiguous()
        mask_true = mask_true.permute(0, 1, 4, 2, 3).contiguous()

        batch = frames.shape[0]
        height = frames.shape[3]
        width = frames.shape[4]

        next_frames = []
        h_t = []
        s_t = []
        d_t = []

        for i in range(self.num_layers):
            zeros = torch.zeros([batch, self.num_hidden[i], height, height]).to(self.configs.device)
            h_t.append(zeros)
            s_t.append(zeros)
            d_t.append(zeros)

        for t in range(self.configs.pre_seq_length + self.configs.aft_seq_length - 1):
            # reverse schedule sampling
       
            if self.configs.reverse_scheduled_sampling == 1:
                if t == 0:
                    net = frames[:, t]
                else:
                    net = mask_true[:, t - 1] * frames[:, t] + (1 - mask_true[:, t - 1]) * x_gen
            else:
                if t < self.configs.pre_seq_length:
                    net = frames[:, t]
                else:
                    net = mask_true[:, t - self.configs.pre_seq_length] * frames[:, t] + \
                            (1 - mask_true[:, t - self.configs.pre_seq_length]) * x_gen

            s_t[0], h_t[0], d_t[0]= self.cell_list[0](net, h_t[0],d_t[0],s_t[0])

            for i in range(1, self.num_layers):
                s_t[i], h_t[i],d_t[i]= self.cell_list[i](s_t[i-1], h_t[i],d_t[i],s_t[i])

            x_gen = self.conv_last(s_t[self.num_layers - 1])
            next_frames.append(x_gen)

        # [length, batch, channel, height, width] -> [batch, length, height, width, channel]
        next_frames = torch.stack(next_frames, dim=0).permute(1, 0, 3, 4, 2).contiguous()
        if kwargs.get('return_loss', True):
            loss = self.MSE_criterion(next_frames, frames_tensor[:, 1:])
        else:
            loss = None

        return next_frames, loss

class LMH_Model(nn.Module):


    def __init__(self, num_layers, num_hidden, configs, **kwargs):
        super(LMH_Model, self).__init__()
        T, C, H, W = configs.in_shape

        self.configs = configs
        self.frame_channel = configs.patch_size * configs.patch_size * C         #帧由1x64x64打乱为16x16x16
        self.num_layers = num_layers
        self.num_hidden = num_hidden
        cell_list = []

        # print('okk')
        # print('num_hidden')
        height = H // configs.patch_size
        width = W // configs.patch_size
        self.MSE_criterion = nn.MSELoss()

        for i in range(num_layers):
            in_channel = self.frame_channel if i == 0 else num_hidden[i - 1]
            cell_list.append(
                LMHCell(in_channel, num_hidden[i], height, height, configs.filter_size,
                                       configs.stride, configs.layer_norm)
            )
        self.cell_list = nn.ModuleList(cell_list)
        self.conv_last = nn.Conv2d(num_hidden[num_layers - 1], self.frame_channel,
                                   kernel_size=1, stride=1, padding=0, bias=False)

    def forward(self, frames_tensor, mask_true, **kwargs):
        # [batch, length, height, width, channel] -> [batch, length, channel, height, width]
        frames = frames_tensor.permute(0, 1, 4, 2, 3).contiguous()
        mask_true = mask_true.permute(0, 1, 4, 2, 3).contiguous()

        batch = frames.shape[0]
        height = frames.shape[3]
        width = frames.shape[4]

        next_frames = []
        h_t = []
        s_t = []
        d_t = []

        for i in range(self.num_layers):
            zeros = torch.zeros([batch, self.num_hidden[i], height, height]).to(self.configs.device)
            h_t.append(zeros)
            s_t.append(zeros)
            d_t.append(zeros)

        for t in range(self.configs.pre_seq_length + self.configs.aft_seq_length - 1):
            # reverse schedule sampling
       
            if self.configs.reverse_scheduled_sampling == 1:
                if t == 0:
                    net = frames[:, t]
                else:
                    net = mask_true[:, t - 1] * frames[:, t] + (1 - mask_true[:, t - 1]) * x_gen
            else:
                if t < self.configs.pre_seq_length:
                    net = frames[:, t]
                else:
                    net = mask_true[:, t - self.configs.pre_seq_length] * frames[:, t] + \
                            (1 - mask_true[:, t - self.configs.pre_seq_length]) * x_gen


            s_t[0], h_t[0], d_t[0]= self.cell_list[0](net, h_t[0],d_t[0],s_t[0])

            for i in range(1, self.num_layers):
                s_t[i], h_t[i],d_t[i]= self.cell_list[i](s_t[i-1], h_t[i],d_t[i],s_t[i])

            x_gen = self.conv_last(s_t[self.num_layers - 1])
            next_frames.append(x_gen)

        # [length, batch, channel, height, width] -> [batch, length, height, width, channel]
        next_frames = torch.stack(next_frames, dim=0).permute(1, 0, 3, 4, 2).contiguous()
        if kwargs.get('return_loss', True):
            loss = self.MSE_criterion(next_frames, frames_tensor[:, 1:])
        else:
            loss = None

        return next_frames, loss

class STCLMH_Model(nn.Module):


    def __init__(self, num_layers, num_hidden, configs, **kwargs):
        super(STCLMH_Model, self).__init__()
        T, C, H, W = configs.in_shape

        self.configs = configs
        self.frame_channel = configs.patch_size * configs.patch_size * C         #帧由1x64x64打乱为16x16x16
        self.num_layers = num_layers
        self.num_hidden = num_hidden
        cell_list = []

        # print('okk')
        # print('num_hidden')
        height = H // configs.patch_size
        width = W // configs.patch_size
        self.MSE_criterion = nn.MSELoss()

        for i in range(num_layers):
            in_channel = self.frame_channel if i == 0 else num_hidden[i - 1]
            cell_list.append(
                STCLMHCell(in_channel, num_hidden[i], height, height, configs.filter_size,
                                       configs.stride, configs.layer_norm,configs.gradient_detach)
            )
        self.cell_list = nn.ModuleList(cell_list)
        self.conv_last = nn.Conv2d(num_hidden[num_layers - 1], self.frame_channel,
                                   kernel_size=1, stride=1, padding=0, bias=False)

    def forward(self, frames_tensor, mask_true, **kwargs):
        # [batch, length, height, width, channel] -> [batch, length, channel, height, width]
        frames = frames_tensor.permute(0, 1, 4, 2, 3).contiguous()
        mask_true = mask_true.permute(0, 1, 4, 2, 3).contiguous()

        batch = frames.shape[0]
        height = frames.shape[3]
        width = frames.shape[4]

        next_frames = []
        h_t = []
        s_t = []
        d_t = []

        for i in range(self.num_layers):
            zeros = torch.zeros([batch, self.num_hidden[i], height, height]).to(self.configs.device)
            h_t.append(zeros)
            s_t.append(zeros)
            d_t.append(zeros)

        for t in range(self.configs.pre_seq_length + self.configs.aft_seq_length - 1):
            # reverse schedule sampling
       
            if self.configs.reverse_scheduled_sampling == 1:
                if t == 0:
                    net = frames[:, t]
                else:
                    net = mask_true[:, t - 1] * frames[:, t] + (1 - mask_true[:, t - 1]) * x_gen
            else:
                if t < self.configs.pre_seq_length:
                    net = frames[:, t]
                else:
                    net = mask_true[:, t - self.configs.pre_seq_length] * frames[:, t] + \
                            (1 - mask_true[:, t - self.configs.pre_seq_length]) * x_gen


            s_t[0], h_t[0], d_t[0]= self.cell_list[0](net, h_t[0],d_t[0],s_t[0])

            for i in range(1, self.num_layers):
                s_t[i], h_t[i],d_t[i]= self.cell_list[i](s_t[i-1], h_t[i],d_t[i],s_t[i])

            x_gen = self.conv_last(s_t[self.num_layers - 1])
            next_frames.append(x_gen)

        # [length, batch, channel, height, width] -> [batch, length, height, width, channel]
        next_frames = torch.stack(next_frames, dim=0).permute(1, 0, 3, 4, 2).contiguous()
        if kwargs.get('return_loss', True):
            loss = self.MSE_criterion(next_frames, frames_tensor[:, 1:])
        else:
            loss = None

        return next_frames, loss
