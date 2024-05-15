import torch.nn as nn

from openstl.models import LIF_Model,PLIF_Model,LMH_Model,TCLIF_Model,STCLIF_Model,STCPLIF_Model,STCLMH_Model,GLIF_Model
from .predrnn import PredRNN
import torch
import time
import torch
import torch.nn as nn
from timm.utils import AverageMeter
from tqdm import tqdm
from spikingjelly.activation_based import functional
from openstl.models import PredRNN_Model
from openstl.utils import (reduce_tensor, reshape_patch, reshape_patch_back,
                           reserve_schedule_sampling_exp, schedule_sampling)

class GLIF(PredRNN):
    r"""GLIF

    Implementation of `Glif: A unified gated leaky integrate-and-fire neuron for spiking neural
    networks.`

    """

    def __init__(self, args, device, steps_per_epoch):
        PredRNN.__init__(self, args, device,  steps_per_epoch)
        self.model = self._build_model(self.args)
        self.model_optim, self.scheduler, self.by_epoch = self._init_optimizer(steps_per_epoch)
        self.criterion = nn.MSELoss()

    def _build_model(self, args):
        num_hidden = [int(x) for x in self.args.num_hidden.split(',')]
        num_layers = len(num_hidden)
        return GLIF_Model(num_layers, num_hidden, args).to(self.device)
    
    def train_one_epoch(self, runner, train_loader, epoch, num_updates, eta=None, **kwargs):
        """Train the model with train_loader."""
        data_time_m = AverageMeter()
        losses_m = AverageMeter()
        self.model.train()
        if self.by_epoch:
            self.scheduler.step(epoch)
        train_pbar = tqdm(train_loader) if self.rank == 0 else train_loader

        end = time.time()
        for batch_x, batch_y in train_pbar:
            data_time_m.update(time.time() - end)
            self.model_optim.zero_grad()

            if not self.args.use_prefetcher:
                batch_x, batch_y = batch_x.to(self.device), batch_y.to(self.device)
            runner.call_hook('before_train_iter')
            
            functional.reset_net(self.model)
            # preprocess
            ims = torch.cat([batch_x, batch_y], dim=1).permute(0, 1, 3, 4, 2).contiguous()
            ims = reshape_patch(ims, self.args.patch_size)
            if self.args.reverse_scheduled_sampling == 1:
                real_input_flag = reserve_schedule_sampling_exp(
                    num_updates, ims.shape[0], self.args)
            else:
                eta, real_input_flag = schedule_sampling(
                    eta, num_updates, ims.shape[0], self.args)

            with self.amp_autocast():
                img_gen, loss = self.model(ims, real_input_flag)

            if not self.dist:
                losses_m.update(loss.item(), batch_x.size(0))

            if self.loss_scaler is not None:
                if torch.any(torch.isnan(loss)) or torch.any(torch.isinf(loss)):
                    raise ValueError("Inf or nan loss value. Please use fp32 training!")
                self.loss_scaler(
                    loss, self.model_optim,
                    clip_grad=self.args.clip_grad, clip_mode=self.args.clip_mode,
                    parameters=self.model.parameters())
            else:
                loss.backward()
                self.clip_grads(self.model.parameters())
                self.model_optim.step()

            torch.cuda.synchronize()
            num_updates += 1

            if self.dist:
                losses_m.update(reduce_tensor(loss), batch_x.size(0))

            if not self.by_epoch:
                self.scheduler.step()
            runner.call_hook('after_train_iter')
            runner._iter += 1

            if self.rank == 0:
                log_buffer = 'train loss: {:.4f}'.format(loss.item())
                log_buffer += ' | data time: {:.4f}'.format(data_time_m.avg)
                train_pbar.set_description(log_buffer)

            end = time.time()  # end for

        if hasattr(self.model_optim, 'sync_lookahead'):
            self.model_optim.sync_lookahead()

        return num_updates, losses_m, eta

class LIF(PredRNN):
    r"""
    LIF model
    """

    def __init__(self, args, device, steps_per_epoch):
        PredRNN.__init__(self, args, device,  steps_per_epoch)
        self.model = self._build_model(self.args)
        self.model_optim, self.scheduler, self.by_epoch = self._init_optimizer(steps_per_epoch)
        self.criterion = nn.MSELoss()

    def _build_model(self, args):
        num_hidden = [int(x) for x in self.args.num_hidden.split(',')]
        num_layers = len(num_hidden)
        return LIF_Model(num_layers, num_hidden, args).to(self.device)
    
    def train_one_epoch(self, runner, train_loader, epoch, num_updates, eta=None, **kwargs):
        """Train the model with train_loader."""
        data_time_m = AverageMeter()
        losses_m = AverageMeter()
        self.model.train()
        if self.by_epoch:
            self.scheduler.step(epoch)
        train_pbar = tqdm(train_loader) if self.rank == 0 else train_loader

        end = time.time()
        for batch_x, batch_y in train_pbar:
            data_time_m.update(time.time() - end)
            self.model_optim.zero_grad()

            if not self.args.use_prefetcher:
                batch_x, batch_y = batch_x.to(self.device), batch_y.to(self.device)
            runner.call_hook('before_train_iter')
            
            functional.reset_net(self.model)
            # preprocess
            ims = torch.cat([batch_x, batch_y], dim=1).permute(0, 1, 3, 4, 2).contiguous()
            ims = reshape_patch(ims, self.args.patch_size)
            if self.args.reverse_scheduled_sampling == 1:
                real_input_flag = reserve_schedule_sampling_exp(
                    num_updates, ims.shape[0], self.args)
            else:
                eta, real_input_flag = schedule_sampling(
                    eta, num_updates, ims.shape[0], self.args)

            with self.amp_autocast():
                img_gen, loss = self.model(ims, real_input_flag)

            if not self.dist:
                losses_m.update(loss.item(), batch_x.size(0))

            if self.loss_scaler is not None:
                if torch.any(torch.isnan(loss)) or torch.any(torch.isinf(loss)):
                    raise ValueError("Inf or nan loss value. Please use fp32 training!")
                self.loss_scaler(
                    loss, self.model_optim,
                    clip_grad=self.args.clip_grad, clip_mode=self.args.clip_mode,
                    parameters=self.model.parameters())
            else:
                loss.backward()
                self.clip_grads(self.model.parameters())
                self.model_optim.step()

            torch.cuda.synchronize()
            num_updates += 1

            if self.dist:
                losses_m.update(reduce_tensor(loss), batch_x.size(0))

            if not self.by_epoch:
                self.scheduler.step()
            runner.call_hook('after_train_iter')
            runner._iter += 1

            if self.rank == 0:
                log_buffer = 'train loss: {:.4f}'.format(loss.item())
                log_buffer += ' | data time: {:.4f}'.format(data_time_m.avg)
                train_pbar.set_description(log_buffer)

            end = time.time()  # end for

        if hasattr(self.model_optim, 'sync_lookahead'):
            self.model_optim.sync_lookahead()

        return num_updates, losses_m, eta
    

class PLIF(PredRNN):
    r"""PLIF model

    Implementation of `Incorporating learnable membrane time constant
    to enhance learning of spiking neural networks.`_.

    """

    def __init__(self, args, device, steps_per_epoch):
        PredRNN.__init__(self, args, device,  steps_per_epoch)
        self.model = self._build_model(self.args)
        self.model_optim, self.scheduler, self.by_epoch = self._init_optimizer(steps_per_epoch)
        self.criterion = nn.MSELoss()

    def _build_model(self, args):
        num_hidden = [int(x) for x in self.args.num_hidden.split(',')]
        num_layers = len(num_hidden)
        return PLIF_Model(num_layers, num_hidden, args).to(self.device)
    
    def train_one_epoch(self, runner, train_loader, epoch, num_updates, eta=None, **kwargs):
        """Train the model with train_loader."""
        data_time_m = AverageMeter()
        losses_m = AverageMeter()
        self.model.train()
        if self.by_epoch:
            self.scheduler.step(epoch)
        train_pbar = tqdm(train_loader) if self.rank == 0 else train_loader

        end = time.time()
        for batch_x, batch_y in train_pbar:
            data_time_m.update(time.time() - end)
            self.model_optim.zero_grad()

            if not self.args.use_prefetcher:
                batch_x, batch_y = batch_x.to(self.device), batch_y.to(self.device)
            runner.call_hook('before_train_iter')
            
            functional.reset_net(self.model)
            # preprocess
            ims = torch.cat([batch_x, batch_y], dim=1).permute(0, 1, 3, 4, 2).contiguous()
            ims = reshape_patch(ims, self.args.patch_size)
            if self.args.reverse_scheduled_sampling == 1:
                real_input_flag = reserve_schedule_sampling_exp(
                    num_updates, ims.shape[0], self.args)
            else:
                eta, real_input_flag = schedule_sampling(
                    eta, num_updates, ims.shape[0], self.args)

            with self.amp_autocast():
                img_gen, loss = self.model(ims, real_input_flag)

            if not self.dist:
                losses_m.update(loss.item(), batch_x.size(0))

            if self.loss_scaler is not None:
                if torch.any(torch.isnan(loss)) or torch.any(torch.isinf(loss)):
                    raise ValueError("Inf or nan loss value. Please use fp32 training!")
                self.loss_scaler(
                    loss, self.model_optim,
                    clip_grad=self.args.clip_grad, clip_mode=self.args.clip_mode,
                    parameters=self.model.parameters())
            else:
                loss.backward()
                self.clip_grads(self.model.parameters())
                self.model_optim.step()

            torch.cuda.synchronize()
            num_updates += 1

            if self.dist:
                losses_m.update(reduce_tensor(loss), batch_x.size(0))

            if not self.by_epoch:
                self.scheduler.step()
            runner.call_hook('after_train_iter')
            runner._iter += 1

            if self.rank == 0:
                log_buffer = 'train loss: {:.4f}'.format(loss.item())
                log_buffer += ' | data time: {:.4f}'.format(data_time_m.avg)
                train_pbar.set_description(log_buffer)

            end = time.time()  # end for

        if hasattr(self.model_optim, 'sync_lookahead'):
            self.model_optim.sync_lookahead()

        return num_updates, losses_m, eta
    
class TCLIF(PredRNN):
    r"""TCLIF model

    Implementation of `Tc-lif: A two-compartment spiking neuron model
    for long-term sequential modelling.`

    """

    def __init__(self, args, device, steps_per_epoch):
        PredRNN.__init__(self, args, device,  steps_per_epoch)
        self.model = self._build_model(self.args)
        self.model_optim, self.scheduler, self.by_epoch = self._init_optimizer(steps_per_epoch)
        self.criterion = nn.MSELoss()

    def _build_model(self, args):
        num_hidden = [int(x) for x in self.args.num_hidden.split(',')]
        num_layers = len(num_hidden)
        return TCLIF_Model(num_layers, num_hidden, args).to(self.device)
    
    def train_one_epoch(self, runner, train_loader, epoch, num_updates, eta=None, **kwargs):
        """Train the model with train_loader."""
        data_time_m = AverageMeter()
        losses_m = AverageMeter()
        self.model.train()
        if self.by_epoch:
            self.scheduler.step(epoch)
        train_pbar = tqdm(train_loader) if self.rank == 0 else train_loader

        end = time.time()
        for batch_x, batch_y in train_pbar:
            data_time_m.update(time.time() - end)
            self.model_optim.zero_grad()

            if not self.args.use_prefetcher:
                batch_x, batch_y = batch_x.to(self.device), batch_y.to(self.device)
            runner.call_hook('before_train_iter')
            
            functional.reset_net(self.model)
            # preprocess
            ims = torch.cat([batch_x, batch_y], dim=1).permute(0, 1, 3, 4, 2).contiguous()
            ims = reshape_patch(ims, self.args.patch_size)
            if self.args.reverse_scheduled_sampling == 1:
                real_input_flag = reserve_schedule_sampling_exp(
                    num_updates, ims.shape[0], self.args)
            else:
                eta, real_input_flag = schedule_sampling(
                    eta, num_updates, ims.shape[0], self.args)

            with self.amp_autocast():
                img_gen, loss = self.model(ims, real_input_flag)

            if not self.dist:
                losses_m.update(loss.item(), batch_x.size(0))

            if self.loss_scaler is not None:
                if torch.any(torch.isnan(loss)) or torch.any(torch.isinf(loss)):
                    raise ValueError("Inf or nan loss value. Please use fp32 training!")
                self.loss_scaler(
                    loss, self.model_optim,
                    clip_grad=self.args.clip_grad, clip_mode=self.args.clip_mode,
                    parameters=self.model.parameters())
            else:
                loss.backward()
                self.clip_grads(self.model.parameters())
                self.model_optim.step()

            torch.cuda.synchronize()
            num_updates += 1

            if self.dist:
                losses_m.update(reduce_tensor(loss), batch_x.size(0))

            if not self.by_epoch:
                self.scheduler.step()
            runner.call_hook('after_train_iter')
            runner._iter += 1

            if self.rank == 0:
                log_buffer = 'train loss: {:.4f}'.format(loss.item())
                log_buffer += ' | data time: {:.4f}'.format(data_time_m.avg)
                train_pbar.set_description(log_buffer)

            end = time.time()  # end for

        if hasattr(self.model_optim, 'sync_lookahead'):
            self.model_optim.sync_lookahead()

        return num_updates, losses_m, eta
    
class LMH(PredRNN):
    r"""LM-H model

    Implementation of `A Progressive Training Framework for Spiking Neural Networks 
    with Learnable Multi-hierarchical Model <https://openreview.net/forum?id=g52tgL8jy6>`_.

    """

    def __init__(self, args, device, steps_per_epoch):
        PredRNN.__init__(self, args, device,  steps_per_epoch)
        self.model = self._build_model(self.args)
        self.model_optim, self.scheduler, self.by_epoch = self._init_optimizer(steps_per_epoch)
        self.criterion = nn.MSELoss()

    def _build_model(self, args):
        num_hidden = [int(x) for x in self.args.num_hidden.split(',')]
        num_layers = len(num_hidden)
        return LMH_Model(num_layers, num_hidden, args).to(self.device)
    
    def train_one_epoch(self, runner, train_loader, epoch, num_updates, eta=None, **kwargs):
        """Train the model with train_loader."""
        data_time_m = AverageMeter()
        losses_m = AverageMeter()
        self.model.train()
        if self.by_epoch:
            self.scheduler.step(epoch)
        train_pbar = tqdm(train_loader) if self.rank == 0 else train_loader

        end = time.time()
        for batch_x, batch_y in train_pbar:
            data_time_m.update(time.time() - end)
            self.model_optim.zero_grad()

            if not self.args.use_prefetcher:
                batch_x, batch_y = batch_x.to(self.device), batch_y.to(self.device)
            runner.call_hook('before_train_iter')
            
            functional.reset_net(self.model)
            # preprocess
            ims = torch.cat([batch_x, batch_y], dim=1).permute(0, 1, 3, 4, 2).contiguous()
            ims = reshape_patch(ims, self.args.patch_size)
            if self.args.reverse_scheduled_sampling == 1:
                real_input_flag = reserve_schedule_sampling_exp(
                    num_updates, ims.shape[0], self.args)
            else:
                eta, real_input_flag = schedule_sampling(
                    eta, num_updates, ims.shape[0], self.args)

            with self.amp_autocast():
                img_gen, loss = self.model(ims, real_input_flag)

            if not self.dist:
                losses_m.update(loss.item(), batch_x.size(0))

            if self.loss_scaler is not None:
                if torch.any(torch.isnan(loss)) or torch.any(torch.isinf(loss)):
                    raise ValueError("Inf or nan loss value. Please use fp32 training!")
                self.loss_scaler(
                    loss, self.model_optim,
                    clip_grad=self.args.clip_grad, clip_mode=self.args.clip_mode,
                    parameters=self.model.parameters())
            else:
                loss.backward()
                self.clip_grads(self.model.parameters())
                self.model_optim.step()

            torch.cuda.synchronize()
            num_updates += 1

            if self.dist:
                losses_m.update(reduce_tensor(loss), batch_x.size(0))

            if not self.by_epoch:
                self.scheduler.step()
            runner.call_hook('after_train_iter')
            runner._iter += 1

            if self.rank == 0:
                log_buffer = 'train loss: {:.4f}'.format(loss.item())
                log_buffer += ' | data time: {:.4f}'.format(data_time_m.avg)
                train_pbar.set_description(log_buffer)

            end = time.time()  # end for

        if hasattr(self.model_optim, 'sync_lookahead'):
            self.model_optim.sync_lookahead()

        return num_updates, losses_m, eta
    

class STCLIF(PredRNN):
    r"""STC-LIF model

    Implementation of "Autaptic Synaptic Circuit Enhances Spatio-temporal Predictive
    Learning of Spiking Neural Networks."

    """

    def __init__(self, args, device, steps_per_epoch):
        PredRNN.__init__(self, args, device,  steps_per_epoch)
        self.model = self._build_model(self.args)
        self.model_optim, self.scheduler, self.by_epoch = self._init_optimizer(steps_per_epoch)
        self.criterion = nn.MSELoss()

    def _build_model(self, args):
        num_hidden = [int(x) for x in self.args.num_hidden.split(',')]
        num_layers = len(num_hidden)
        return STCLIF_Model(num_layers, num_hidden, args).to(self.device)
    
    def train_one_epoch(self, runner, train_loader, epoch, num_updates, eta=None, **kwargs):
        """Train the model with train_loader."""
        data_time_m = AverageMeter()
        losses_m = AverageMeter()
        self.model.train()
        if self.by_epoch:
            self.scheduler.step(epoch)
        train_pbar = tqdm(train_loader) if self.rank == 0 else train_loader

        end = time.time()
        for batch_x, batch_y in train_pbar:
            data_time_m.update(time.time() - end)
            self.model_optim.zero_grad()

            if not self.args.use_prefetcher:
                batch_x, batch_y = batch_x.to(self.device), batch_y.to(self.device)
            runner.call_hook('before_train_iter')
            
            functional.reset_net(self.model)
            # preprocess
            ims = torch.cat([batch_x, batch_y], dim=1).permute(0, 1, 3, 4, 2).contiguous()
            ims = reshape_patch(ims, self.args.patch_size)
            if self.args.reverse_scheduled_sampling == 1:
                real_input_flag = reserve_schedule_sampling_exp(
                    num_updates, ims.shape[0], self.args)
            else:
                eta, real_input_flag = schedule_sampling(
                    eta, num_updates, ims.shape[0], self.args)

            with self.amp_autocast():
                img_gen, loss = self.model(ims, real_input_flag)

            if not self.dist:
                losses_m.update(loss.item(), batch_x.size(0))

            if self.loss_scaler is not None:
                if torch.any(torch.isnan(loss)) or torch.any(torch.isinf(loss)):
                    raise ValueError("Inf or nan loss value. Please use fp32 training!")
                self.loss_scaler(
                    loss, self.model_optim,
                    clip_grad=self.args.clip_grad, clip_mode=self.args.clip_mode,
                    parameters=self.model.parameters())
            else:
                loss.backward()
                self.clip_grads(self.model.parameters())
                self.model_optim.step()

            torch.cuda.synchronize()
            num_updates += 1

            if self.dist:
                losses_m.update(reduce_tensor(loss), batch_x.size(0))

            if not self.by_epoch:
                self.scheduler.step()
            runner.call_hook('after_train_iter')
            runner._iter += 1

            if self.rank == 0:
                log_buffer = 'train loss: {:.4f}'.format(loss.item())
                log_buffer += ' | data time: {:.4f}'.format(data_time_m.avg)
                train_pbar.set_description(log_buffer)

            end = time.time()  # end for

        if hasattr(self.model_optim, 'sync_lookahead'):
            self.model_optim.sync_lookahead()

        return num_updates, losses_m, eta
    

class STCPLIF(PredRNN):
    r"""STC-PLIF model

    Implementation of "Autaptic Synaptic Circuit Enhances Spatio-temporal Predictive
    Learning of Spiking Neural Networks."

    """

    def __init__(self, args, device, steps_per_epoch):
        PredRNN.__init__(self, args, device,  steps_per_epoch)
        self.model = self._build_model(self.args)
        self.model_optim, self.scheduler, self.by_epoch = self._init_optimizer(steps_per_epoch)
        self.criterion = nn.MSELoss()

    def _build_model(self, args):
        num_hidden = [int(x) for x in self.args.num_hidden.split(',')]
        num_layers = len(num_hidden)
        return STCPLIF_Model(num_layers, num_hidden, args).to(self.device)
    
    def train_one_epoch(self, runner, train_loader, epoch, num_updates, eta=None, **kwargs):
        """Train the model with train_loader."""
        data_time_m = AverageMeter()
        losses_m = AverageMeter()
        self.model.train()
        if self.by_epoch:
            self.scheduler.step(epoch)
        train_pbar = tqdm(train_loader) if self.rank == 0 else train_loader

        end = time.time()
        for batch_x, batch_y in train_pbar:
            data_time_m.update(time.time() - end)
            self.model_optim.zero_grad()

            if not self.args.use_prefetcher:
                batch_x, batch_y = batch_x.to(self.device), batch_y.to(self.device)
            runner.call_hook('before_train_iter')
            
            functional.reset_net(self.model)
            # preprocess
            ims = torch.cat([batch_x, batch_y], dim=1).permute(0, 1, 3, 4, 2).contiguous()
            ims = reshape_patch(ims, self.args.patch_size)
            if self.args.reverse_scheduled_sampling == 1:
                real_input_flag = reserve_schedule_sampling_exp(
                    num_updates, ims.shape[0], self.args)
            else:
                eta, real_input_flag = schedule_sampling(
                    eta, num_updates, ims.shape[0], self.args)

            with self.amp_autocast():
                img_gen, loss = self.model(ims, real_input_flag)

            if not self.dist:
                losses_m.update(loss.item(), batch_x.size(0))

            if self.loss_scaler is not None:
                if torch.any(torch.isnan(loss)) or torch.any(torch.isinf(loss)):
                    raise ValueError("Inf or nan loss value. Please use fp32 training!")
                self.loss_scaler(
                    loss, self.model_optim,
                    clip_grad=self.args.clip_grad, clip_mode=self.args.clip_mode,
                    parameters=self.model.parameters())
            else:
                loss.backward()
                self.clip_grads(self.model.parameters())
                self.model_optim.step()

            torch.cuda.synchronize()
            num_updates += 1

            if self.dist:
                losses_m.update(reduce_tensor(loss), batch_x.size(0))

            if not self.by_epoch:
                self.scheduler.step()
            runner.call_hook('after_train_iter')
            runner._iter += 1

            if self.rank == 0:
                log_buffer = 'train loss: {:.4f}'.format(loss.item())
                log_buffer += ' | data time: {:.4f}'.format(data_time_m.avg)
                train_pbar.set_description(log_buffer)

            end = time.time()  # end for

        if hasattr(self.model_optim, 'sync_lookahead'):
            self.model_optim.sync_lookahead()

        return num_updates, losses_m, eta
    
class STCLMH(PredRNN):
    r"""STC-LM-H model

    Implementation of "Autaptic Synaptic Circuit Enhances Spatio-temporal Predictive
    Learning of Spiking Neural Networks."

    """

    def __init__(self, args, device, steps_per_epoch):
        PredRNN.__init__(self, args, device,  steps_per_epoch)
        self.model = self._build_model(self.args)
        self.model_optim, self.scheduler, self.by_epoch = self._init_optimizer(steps_per_epoch)
        self.criterion = nn.MSELoss()

    def _build_model(self, args):
        num_hidden = [int(x) for x in self.args.num_hidden.split(',')]
        num_layers = len(num_hidden)
        return STCLMH_Model(num_layers, num_hidden, args).to(self.device)
    
    def train_one_epoch(self, runner, train_loader, epoch, num_updates, eta=None, **kwargs):
        """Train the model with train_loader."""
        data_time_m = AverageMeter()
        losses_m = AverageMeter()
        self.model.train()
        if self.by_epoch:
            self.scheduler.step(epoch)
        train_pbar = tqdm(train_loader) if self.rank == 0 else train_loader

        end = time.time()
        for batch_x, batch_y in train_pbar:
            data_time_m.update(time.time() - end)
            self.model_optim.zero_grad()

            if not self.args.use_prefetcher:
                batch_x, batch_y = batch_x.to(self.device), batch_y.to(self.device)
            runner.call_hook('before_train_iter')
            
            functional.reset_net(self.model)
            # preprocess
            ims = torch.cat([batch_x, batch_y], dim=1).permute(0, 1, 3, 4, 2).contiguous()
            ims = reshape_patch(ims, self.args.patch_size)
            if self.args.reverse_scheduled_sampling == 1:
                real_input_flag = reserve_schedule_sampling_exp(
                    num_updates, ims.shape[0], self.args)
            else:
                eta, real_input_flag = schedule_sampling(
                    eta, num_updates, ims.shape[0], self.args)

            with self.amp_autocast():
                img_gen, loss = self.model(ims, real_input_flag)

            if not self.dist:
                losses_m.update(loss.item(), batch_x.size(0))

            if self.loss_scaler is not None:
                if torch.any(torch.isnan(loss)) or torch.any(torch.isinf(loss)):
                    raise ValueError("Inf or nan loss value. Please use fp32 training!")
                self.loss_scaler(
                    loss, self.model_optim,
                    clip_grad=self.args.clip_grad, clip_mode=self.args.clip_mode,
                    parameters=self.model.parameters())
            else:
                loss.backward()
                self.clip_grads(self.model.parameters())
                self.model_optim.step()

            torch.cuda.synchronize()
            num_updates += 1

            if self.dist:
                losses_m.update(reduce_tensor(loss), batch_x.size(0))

            if not self.by_epoch:
                self.scheduler.step()
            runner.call_hook('after_train_iter')
            runner._iter += 1

            if self.rank == 0:
                log_buffer = 'train loss: {:.4f}'.format(loss.item())
                log_buffer += ' | data time: {:.4f}'.format(data_time_m.avg)
                train_pbar.set_description(log_buffer)

            end = time.time()  # end for

        if hasattr(self.model_optim, 'sync_lookahead'):
            self.model_optim.sync_lookahead()

        return num_updates, losses_m, eta
