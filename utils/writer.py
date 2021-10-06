import os
import time
import torch
import json
from glob import glob
from torch.utils.tensorboard import SummaryWriter


class Writer:
    def __init__(self, args=None):
        self.args = args

        if self.args is not None:
            tmp_log_list = glob(os.path.join(args.out_dir, 'log*'))
            if len(tmp_log_list) == 0:
                self.log_file = os.path.join(
                    args.out_dir, 'log_{:s}.txt'.format(
                        time.strftime('%Y-%m-%d-%H-%M-%S', time.localtime())))
            else:
                self.log_file = tmp_log_list[0]
        self.writer = SummaryWriter(os.path.join(
            args.out_dir, "board", f"{args.dataset}_{args.epochs}_{time.strftime('%Y-%m-%d-%H-%M-%S', time.localtime())}")
        )

    def print_info(self, info):
        message = 'Epoch: {}/{}, Duration: {:.3f}s, Train Loss: {:.4f}, Test Loss: {:.4f}, C1 test loss {:.4f}'.format(
            info['current_epoch'], info['epochs'], info['t_duration'],
            info['train_loss'], info['test_loss'], info['c1_test']
        )
        with open(self.log_file, 'a') as log_file:
            log_file.write('{:s}\n'.format(message))
        print(message)
        self.writer.add_scalar(
            'Loss/train', info['train_loss'], info['current_epoch']
        )
        self.writer.add_scalar(
            'Loss/test', info['test_loss'], info['current_epoch']
        )
        if 'c1_test' in info:
            self.writer.add_scalar(
                'Loss/test_c1', info['c1_test'], info['current_epoch']
            )
            self.writer.add_scalar(
                'Loss/train_c1', info['c1_train'], info['current_epoch']
            )

    def save_checkpoint(self, model, optimizer, scheduler, epoch):
        torch.save(
            {
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'scheduler_state_dict': scheduler.state_dict(),
            },
            os.path.join(self.args.checkpoints_dir,
                         'checkpoint_{:03d}.pt'.format(epoch)))

    def add_graph(self, model, input):
        print(input.size())
        self.writer.add_graph(
            model=model, input_to_model=input, verbose=True)
