import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models
# from torch.cuda.amp import autocast, GradScaler

import os
import contextlib

import pandas as pd
import numpy as np

from .scmatch_utils import consistency_loss, Get_Scalar, ce_loss
from .scmatch_utils import AverageMeter, accuracy
from .scmatch_utils import contrast_loss_clust
from .scmatch_utils import exp_rampup, linear_rampup, step_rampup

ndim_feat = 32


class CatMatch:
    def __init__(self, net_builder, num_classes, ema_m, T, p_cutoff, lambda_u, lambda_c,
                 num_train_iter, min_clust_K, max_clust_K, method_ramp_K="linear", rampup_ratio=0.8,
                 clust_cutoff=0.8, clust_cont_temp=1.0, cluster_from_iter=0, flag_std_cont=False,
                 hard_label=True, t_fn=None, p_fn=None, it=0,
                 num_eval_iter=1000, tb_log=None, logger=None, csv_path=None, device_id=0):
        super(CatMatch, self).__init__()

        # momentum update param
        self.loader = {}
        self.num_classes = num_classes
        self.ema_m = ema_m
        self.device_id = device_id

        self.train_model = net_builder(num_classes=num_classes)
        self.eval_model = net_builder(num_classes=num_classes)
        self.num_eval_iter = num_eval_iter
        self.num_train_iter = num_train_iter
        self.t_fn = Get_Scalar(T)  # temperature params function
        self.p_fn = Get_Scalar(p_cutoff)  # confidence cutoff function
        self.lambda_u = lambda_u
        self.tb_log = tb_log
        self.use_hard_label = hard_label

        self.optimizer = None
        self.scheduler = None

        self.it = 0

        self.logger = logger
        self.print_fn = print if logger is None else logger.info

        self.csv_path = csv_path
        self.csv_path_mean = os.path.splitext(os.path.realpath(csv_path))[0] + "_mean.csv"

        for param_q, param_k in zip(self.train_model.parameters(), self.eval_model.parameters()):
            param_k.data.copy_(param_q.detach().data)  # initialize
            param_k.requires_grad = False  # not update by gradient for eval_net

        self.eval_model.eval()

        # cluster related params
        self.min_k = min_clust_K
        self.max_k = max_clust_K
        if method_ramp_K == "linear":
            self.func_rampup = linear_rampup(int(self.num_train_iter * rampup_ratio))
        else:
            self.func_rampup = exp_rampup(int(self.num_train_iter * rampup_ratio))

        self.lambda_c = lambda_c
        self.clust_cutoff = clust_cutoff
        self.clust_cont_temp = clust_cont_temp
        self.cluster_from_iter = cluster_from_iter
        self.flag_std_cont = flag_std_cont

    @torch.no_grad()
    def _eval_model_update(self):
        """
        Momentum update of evaluation model (exponential moving average)
        """
        # if hasattr(self.train_model, "module"):
        #     tmp_para = self.train_model.module.parameters()
        # else:
        #     tmp_para = self.train_model.parameters()

        # for param_train, param_eval in zip(tmp_para, self.eval_model.parameters()):
        for param_train, param_eval in zip(self.train_model.parameters(), self.eval_model.parameters()):
            alpha = min(1 - 1 / (self.it + 1), self.ema_m)
            # alpha = self.ema_m
            param_eval.copy_(param_eval * alpha + param_train.detach() * (1 - alpha))

        for buffer_train, buffer_eval in zip(self.train_model.buffers(), self.eval_model.buffers()):
            buffer_eval.copy_(buffer_train)

    def set_data_loader(self, loader_dict):
        self.loader_dict = loader_dict
        self.print_fn(f'[!] data loader keys: {self.loader_dict.keys()}')

    def set_optimizer(self, optimizer, scheduler=None):
        self.optimizer = optimizer
        self.scheduler = scheduler

    def train(self, args, logger=None):
        """
        Train function of CatMatch.
        From data_loader, it inference training data, computes losses, and update the networks.
        """
        ngpus_per_node = torch.cuda.device_count()

        # lb: labeled, ulb: unlabeled
        self.train_model.train()
        # if self.eval_model is not None:
        #     self.eval_model.train()

        # for gpu profiling
        start_batch = torch.cuda.Event(enable_timing=True)
        end_batch = torch.cuda.Event(enable_timing=True)
        start_run = torch.cuda.Event(enable_timing=True)
        end_run = torch.cuda.Event(enable_timing=True)

        start_batch.record()
        best_eval_acc, best_it = 0.0, 0
        best_eval_acc_ema, best_it_ema = 0.0, 0

        amp_cm = contextlib.nullcontext

        # various metrics
        meter_loss_x = AverageMeter()
        meter_loss_u = AverageMeter()
        meter_loss_c = AverageMeter()

        meter_amount_high_ulbs = AverageMeter()
        meter_amount_acc_ulbs = AverageMeter()
        meter_ratio_quantity = AverageMeter()
        meter_ratio_quality = AverageMeter()

        for (x_lb, y_lb), (x_ulb_w, x_ulb_s, y_ulb) in zip(self.loader_dict['train_lb'], self.loader_dict['train_ulb']):

            # prevent the training iterations exceed args.num_train_iter
            if self.it > args.num_train_iter:
                break

            end_batch.record()
            torch.cuda.synchronize()
            start_run.record()

            num_lb = x_lb.shape[0]
            num_ulb = x_ulb_w.shape[0]
            assert num_ulb == x_ulb_s.shape[0]

            x_lb, x_ulb_w, x_ulb_s = x_lb.cuda(args.gpu), x_ulb_w.cuda(args.gpu), x_ulb_s.cuda(args.gpu)
            y_lb = y_lb.cuda(args.gpu)
            y_ulb = y_ulb.cuda(args.gpu)

            inputs = torch.cat((x_lb, x_ulb_w, x_ulb_s))

            # inference and calculate sup/unsup losses
            with amp_cm():
                logits, feats = self.train_model(inputs, ood_test=True)
                logits_x_lb = logits[:num_lb]
                logits_x_ulb_w, logits_x_ulb_s = logits[num_lb:].chunk(2)
                del logits
                feats_ulb_w, feats_ulb_s = feats[num_lb:].chunk(2)
                del feats

                T = self.t_fn(self.it)
                p_cutoff = self.p_fn(self.it)

                sup_loss = ce_loss(logits_x_lb, y_lb, reduction='mean')
                unsup_loss, mask, num_high, num_high_accurate, dist_ulb_all, dist_ulb_high = consistency_loss(
                    logits_x_ulb_w,
                    logits_x_ulb_s, y_ulb, 'ce', T, p_cutoff,
                    use_hard_labels=self.use_hard_label)

                curr_K = self.min_k + np.floor((self.max_k - self.min_k) * self.func_rampup(self.it))
                curr_K = step_rampup(int(curr_K))
                ctr_loss, y_cluster = contrast_loss_clust(logits_x_ulb_w, logits_x_ulb_s, feats_ulb_w,
                                               curr_K, n_feats=ndim_feat, temperature=self.clust_cont_temp,
                                               clust_cutoff=self.clust_cutoff, device_id=self.device_id)


                total_loss = sup_loss + self.lambda_u * unsup_loss + self.lambda_c * ctr_loss


            total_loss.backward()
            self.optimizer.step()
            self.scheduler.step()
            self.train_model.zero_grad()

            with torch.no_grad():
                self._eval_model_update()

            end_run.record()
            torch.cuda.synchronize()

            # tensorboard_dict update
            tb_dict = {}
            tb_dict['train/sup_loss'] = sup_loss.detach().item()
            tb_dict['train/unsup_loss'] = unsup_loss.detach().item()
            tb_dict['train/contr_loss'] = ctr_loss.detach().item()
            tb_dict['train/total_loss'] = total_loss.detach().item()
            # tb_dict['train/mask_ratio'] = 1.0 - mask.detach().item()
            tb_dict['train/pseudo_amount'] = num_high.detach().item()
            tb_dict['train/pseudo_amount_acc'] = num_high_accurate.detach().item()
            tb_dict['train/quantity'] = mask.detach().item()
            tb_dict['train/quality'] = 0.0 if tb_dict['train/pseudo_amount'] == 0 else tb_dict[
                                                                                           'train/pseudo_amount_acc'] / \
                                                                                       tb_dict['train/pseudo_amount']
            tb_dict['lr'] = self.optimizer.param_groups[0]['lr']
            tb_dict['train/prefecth_time'] = start_batch.elapsed_time(end_batch) / 1000.
            tb_dict['train/run_time'] = start_run.elapsed_time(end_run) / 1000.
            tb_dict['eval/acc'], tb_dict['eval/acc-ema'] = 0.0, 0.0

            # dist
            tb_dict['dist/all'], tb_dict['dist/high'] = "", ""
            if dist_ulb_all is not None:
                tb_dict['dist/all'] = ",".join([str(ss) for ss in dist_ulb_all])
            if dist_ulb_high is not None:
                tb_dict['dist/high'] = ",".join([str(ss) for ss in dist_ulb_high])

            # update meters
            meter_loss_x.update(tb_dict['train/sup_loss'])
            meter_loss_u.update(tb_dict['train/unsup_loss'])
            meter_loss_c.update(tb_dict['train/contr_loss'])
            meter_amount_high_ulbs.update(tb_dict['train/pseudo_amount'])
            meter_amount_acc_ulbs.update(tb_dict['train/pseudo_amount_acc'])
            meter_ratio_quantity.update(tb_dict['train/quantity'])
            meter_ratio_quality.update(tb_dict['train/quality'])

            if self.it % self.num_eval_iter == 0:
                eval_dict = self.evaluate(args=args)
                tb_dict.update(eval_dict)
                save_path = os.path.join(args.save_dir, args.save_name)

                if tb_dict['eval/acc'] > best_eval_acc:
                    best_eval_acc = tb_dict['eval/acc']
                    best_it = self.it

                if tb_dict['eval/acc-ema'] > best_eval_acc_ema:
                    best_eval_acc_ema = tb_dict['eval/acc-ema']
                    best_it_ema = self.it

                # self.print_fn("-"*100)
                self.print_fn(
                    " >> TRAIN  Iter:{} lossX:{:.3f} lossU:{:.3f} lossC:{:.3f} correctU:{:.2f}/{:.2f} Pseudo:{:.3f}/{:.3f} K:{} R:{:.3f} T:{:.3f}".format(
                        self.it, meter_loss_x.avg, meter_loss_u.avg, meter_loss_c.avg,
                        tb_dict["train/pseudo_amount_acc"], tb_dict['train/pseudo_amount'],
                        tb_dict["train/quantity"], tb_dict["train/quality"], curr_K, tb_dict['lr'],
                        tb_dict["train/run_time"]))
                self.print_fn(" >> [TEST] Iter:{} Acc/Acc-EMA:{:.3f}/{:.3f} Best:{:.3f}/{} Best-EMA:{:.3f}/{}".format(
                    self.it, tb_dict['eval/acc'], tb_dict['eval/acc-ema'],
                    best_eval_acc, best_it, best_eval_acc_ema, best_it_ema))

            if not args.multiprocessing_distributed or \
                    (args.multiprocessing_distributed and args.rank % ngpus_per_node == 0):

                if self.csv_path_mean is not None:
                    if self.it % self.num_eval_iter == 0:
                        # get the mean performance
                        tmp_dict = {}
                        tmp_dict['train/sup_loss'] = meter_loss_x.avg
                        tmp_dict['train/unsup_loss'] = meter_loss_u.avg
                        tmp_dict['train/contr_loss'] = meter_loss_c.avg
                        tmp_dict['train/pseudo_amount'] = meter_amount_high_ulbs.avg
                        tmp_dict['train/pseudo_amount_acc'] = meter_amount_acc_ulbs.avg
                        tmp_dict['train/quantity'] = meter_ratio_quantity.avg
                        tmp_dict['train/quality'] = meter_ratio_quality.avg
                        tmp_dict['test/acc'] = tb_dict['eval/acc']
                        tmp_dict['test/acc-ema'] = tb_dict['eval/acc-ema']
                        tmp_dict['dist/all'] = tb_dict['dist/all']
                        tmp_dict['dist/high'] = tb_dict['dist/high']

                        if self.it > 0:
                            meter_loss_x.reset()
                            meter_loss_u.reset()
                            meter_loss_c.reset()
                            meter_amount_high_ulbs.reset()
                            meter_amount_acc_ulbs.reset()
                            meter_ratio_quantity.reset()
                            meter_ratio_quality.reset()

            self.it += 1
            del tb_dict
            start_batch.record()

            if self.it > 2 ** 19:
                self.num_eval_iter = 1000

        eval_dict = self.evaluate(args=args)
        eval_dict.update({'eval/best_acc': best_eval_acc, 'eval/best_it': best_it})
        eval_dict.update({'eval/best_acc_ema': best_eval_acc_ema, 'eval/best_it_ema': best_it_ema})
        return eval_dict

    @torch.no_grad()
    def evaluate(self, eval_loader=None, args=None):
        use_ema = hasattr(self, 'eval_model')
        stu_model = self.train_model
        stu_model.eval()
        tea_model = self.eval_model if use_ema else None
        if tea_model is not None:
            tea_model.eval()

        top1_meter = AverageMeter()
        ema_top1_meter = AverageMeter()

        if eval_loader is None:
            eval_loader = self.loader_dict['eval']

        for x, y in eval_loader:
            ims, lbs = x.cuda(args.gpu), y.cuda(args.gpu)

            logits = stu_model(ims)
            scores = torch.softmax(logits, dim=1)
            top1, top5 = accuracy(scores, lbs, (1, 5))
            top1_meter.update(top1.item())

            if tea_model is not None:
                logits = tea_model(ims)
                scores = torch.softmax(logits, dim=1)
                top1, top5 = accuracy(scores, lbs, (1, 5))
                ema_top1_meter.update(top1.item())

        stu_model.train()
        return {'eval/acc': top1_meter.avg, 'eval/acc-ema': ema_top1_meter.avg}

    def save_model(self, save_name, save_path):
        save_filename = os.path.join(save_path, save_name)
        train_model = self.train_model.module if hasattr(self.train_model, 'module') else self.train_model
        eval_model = self.eval_model.module if hasattr(self.eval_model, 'module') else self.eval_model
        torch.save({'train_model': train_model.state_dict(),
                    'eval_model': eval_model.state_dict(),
                    'optimizer': self.optimizer.state_dict(),
                    'scheduler': self.scheduler.state_dict(),
                    'it': self.it}, save_filename)

        self.print_fn(f"[!] model saved: {save_filename}")

    def load_model(self, load_path):
        checkpoint = torch.load(load_path)

        train_model = self.train_model.module if hasattr(self.train_model, 'module') else self.train_model
        eval_model = self.eval_model.module if hasattr(self.eval_model, 'module') else self.eval_model

        for key in checkpoint.keys():
            if hasattr(self, key) and getattr(self, key) is not None:
                if 'train_model' in key:
                    train_model.load_state_dict(checkpoint[key])
                elif 'eval_model' in key:
                    eval_model.load_state_dict(checkpoint[key])
                elif key == 'it':
                    self.it = checkpoint[key]
                elif key == 'scheduler':
                    self.scheduler.load_state_dict(checkpoint[key])
                elif key == 'optimizer':
                    self.optimizer.load_state_dict(checkpoint[key])
                else:
                    getattr(self, key).load_state_dict(checkpoint[key])
                self.print_fn(f"Check Point Loading: {key} is LOADED")
            else:
                self.print_fn(f"Check Point Loading: {key} is **NOT** LOADED")


if __name__ == "__main__":
    pass
