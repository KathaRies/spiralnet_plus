import time
import os
import numpy as np
import torch
import torch.nn.functional as F
import openmesh as om


def c1_loss(label, pred) -> float:
    # tf.experimental.numpy.diff(pred, n=2, axis=0)
    h_loss = (pred[:, 1:]-pred[:, :-1])[:, 1:] - \
        (pred[:, 1:]-pred[:, :-1])[:, :-1]
    h_loss = abs(h_loss[:, 1::2])
    # tf.experimental.numpy.diff(pred, n=2, axis=1)
    v_loss = (pred[:, :, 1:]-pred[:, :, :-1])[:, :, 1:] - \
        (pred[:, :, 1:]-pred[:, :, :-1])[:, :, :-1]
    v_loss = abs(v_loss[:, :, 1::2])
    # tf.math.reduce_sum(h_loss) + tf.math.reduce_sum(v_loss)
    return torch.sum(h_loss) + torch.sum(v_loss)


def run(model, train_loader, test_loader, epochs, optimizer, scheduler, writer,
        device):
    train_losses, test_losses = [], []

    for epoch in range(1, epochs + 1):
        t = time.time()
        train_loss = train(model, optimizer, train_loader, device)
        t_duration = time.time() - t
        test_loss = test(model, test_loader, device)
        scheduler.step()
        info = {
            'current_epoch': epoch,
            'epochs': epochs,
            'train_loss': train_loss,
            'test_loss': test_loss,
            't_duration': t_duration
        }

        writer.print_info(info)
        writer.save_checkpoint(model, optimizer, scheduler, epoch)


def train(model, optimizer, loader, device):
    model.train()

    total_loss = 0
    for data in loader:
        optimizer.zero_grad()
        x = data.x.to(device)
        y = data.y.to(device)
        out = model(x)
        # F.mse_loss(out, y)  # F.l1_loss(out, y, reduction='mean')
        # (0.01*c1_loss(y, out) + F.l1_loss(out, y, reduction='mean'))
        loss = F.l1_loss(out, y, reduction='mean')
        loss.backward()
        total_loss += loss.item()
        optimizer.step()
    return total_loss / len(loader)


def test(model, loader, device):
    model.eval()

    total_loss = 0
    with torch.no_grad():
        for i, data in enumerate(loader):
            x = data.x.to(device)
            y = data.y.to(device)
            pred = model(x)
            # F.l1_loss(pred, y, reduction='mean')
            # F.mse_loss(pred, y)
            total_loss += F.l1_loss(pred, y, reduction='mean')
            #(0.01*c1_loss(y, pred) + F.l1_loss(pred, y, reduction='mean'))
    return total_loss / len(loader)


def eval_error(model, test_loader, device, meshdata, out_dir):
    model.eval()

    errors = []
    mean = meshdata.mean
    std = meshdata.std
    with torch.no_grad():
        for i, data in enumerate(test_loader):
            x = data.x.to(device)
            y = data.y.to(device)
            pred = model(x)
            num_graphs = data.num_graphs
            reshaped_pred = (pred.view(num_graphs, -1, 3).cpu() * std) + mean
            reshaped_y = (y.view(num_graphs, -1, 3).cpu() * std) + mean

            reshaped_pred *= 1000
            reshaped_y *= 1000

            tmp_error = torch.sqrt(
                torch.sum((reshaped_pred - reshaped_y)**2,
                          dim=2))  # [num_graphs, num_nodes]
            errors.append(tmp_error)
            if i == 0:
                pred_mesh = om.TriMesh(
                    points=pred[0].numpy(),
                    face_vertex_indices=data.face[0].numpy().transpose()
                )
                om.write_mesh(mesh=pred_mesh, filename="test_result.obj")
                label_mesh = om.TriMesh(
                    points=y[0].numpy(),
                    face_vertex_indices=data.face[0].numpy().transpose()
                )
                om.write_mesh(mesh=label_mesh, filename="test_label.obj")
        new_errors = torch.cat(errors, dim=0)  # [n_total_graphs, num_nodes]

        mean_error = new_errors.view((-1, )).mean()
        std_error = new_errors.view((-1, )).std()
        median_error = new_errors.view((-1, )).median()

    message = 'Error: {:.3f}+{:.3f} | {:.3f}'.format(mean_error, std_error,
                                                     median_error)

    out_error_fp = out_dir + '/euc_errors.txt'
    with open(out_error_fp, 'a') as log_file:
        log_file.write('{:s}\n'.format(message))
    print(message)
