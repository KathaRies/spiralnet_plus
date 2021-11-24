from enum import Enum
import time
import os
import numpy as np
import torch
import torch.nn.functional as F
import openmesh as om


class Loss(Enum):
    MSE = "mse"
    L1 = "l1"
    MSE_C1 = "mse_c1"
    L1_C1 = "l1_c1"
    COMBI = "combi"

    def get_loss(self):
        if self == Loss.MSE:
            return F.mse_loss
        elif self == Loss.L1:
            return F.l1_loss
        elif self == Loss.MSE_C1:
            return mse_c1
        elif self == Loss.L1_C1:
            return l1_c1
        elif self == Loss.COMBI:
            return combined_loss
        else:
            raise ValueError("Unkown loss")


WEIGHT = 0.01

# mask = torch.ones((9, 9, 1))
# m = 3
# mask[m:-m, m:-m] = 0
# mask = mask.flatten(end_dim=-2)


def combined_loss(x, y, mask):
    mask = mask.type(torch.bool)
    # with x only the hole nad y the whole
    edge_penalty = 100
    f = torch.zeros_like(x)
    for e, elem in enumerate(f):
        f[e] = y[e, ~mask[e, :, 0], :]
    distance = F.mse_loss(x, f)
    c = y.clone()
    for e, elem in enumerate(y):
        c[e, ~mask[e, :, 0], :] = x[e]
    # F.mse_loss(x,y)
    curvation = c1_loss(c, y)
    return curvation * distance

# def combined_loss(x, y, mask=mask):
#     edge_penalty = 100
#     distance = torch.mean(
#         (mask*edge_penalty) * F.mse_loss(x, y)
#     )
#     # F.mse_loss(x,y)
#     curvation = c1_loss(x, y)
#     return distance*curvation


def mse_c1(x, y):
    return F.mse_loss(x, y)+WEIGHT*c1_loss(x, y)


def l1_c1(x, y):
    return F.l1_loss(x, y)+WEIGHT*c1_loss(x, y)


LOSS = F.mse_loss


def c1_loss(pred, label) -> float:
    patch_size = int(np.sqrt(pred.shape[1]))
    degree = 3 if patch_size-1 % 3 == 0 else 2
    assert degree == 2
    start = degree-1
    pred = torch.reshape(pred, (pred.size()[0], patch_size, patch_size, 3))
    h_loss = (pred[:, 1:]-pred[:, :-1])[:, 1:] - \
        (pred[:, 1:]-pred[:, :-1])[:, :-1]
    h_loss = abs(h_loss[:, start::degree])
    # tf.experimental.numpy.diff(pred, n=2, axis=1)
    v_loss = (pred[:, :, 1:]-pred[:, :, :-1])[:, :, 1:] - \
        (pred[:, :, 1:]-pred[:, :, :-1])[:, :, :-1]
    v_loss = abs(v_loss[:, :, start::degree])
    # tf.math.reduce_sum(h_loss) + tf.math.reduce_sum(v_loss)
    # torch.mean()
    return torch.mean(torch.sum(input=h_loss, dim=(1, 2, 3), keepdim=True) + torch.sum(v_loss, dim=(1, 2, 3), keepdim=True))


def c1_eval(model, loader, use_mask, device) -> float:
    model.eval()

    total_loss = 0
    with torch.no_grad():
        for i, data in enumerate(loader):
            x = data.x.to(device)
            y = data.y.to(device)
            if use_mask:
                mask = data.mask.to(device)
                x = torch.cat((x, mask), -1)
            pred = model(x)
            c = y.clone()
            mask = mask.type(torch.bool)
            for e, elem in enumerate(y):
                c[e, ~mask[e, :, 0], :] = pred[e]
            l = c1_loss(c, y)
            total_loss += l
    return total_loss / len(loader)


def run(model, train_loader, test_loader, epochs, optimizer, scheduler, writer,
        device, use_mask, loss):

    for epoch in range(1, epochs + 1):
        t = time.time()
        train_loss = train(model, optimizer, train_loader,
                           device, use_mask, in_loss=loss)
        t_duration = time.time() - t
        test_loss = test(model, test_loader, device, use_mask, in_loss=loss)
        scheduler.step()
        info = {
            'current_epoch': epoch,
            'epochs': epochs,
            'train_loss': train_loss,
            'test_loss': test_loss,
            't_duration': t_duration,
            'c1_test': c1_eval(model, test_loader, use_mask, device),
            'mse_test': test_mse(model, test_loader, device, use_mask)
        }

        writer.print_info(info)
    writer.save_checkpoint(model, optimizer, scheduler, epoch)
    return info


def train(model, optimizer, loader, device, use_mask, in_loss):
    model.train()

    total_loss = 0
    for i, data in enumerate(loader):
        optimizer.zero_grad()
        x = data.x.to(device)
        y = data.y.to(device)
        if use_mask:
            mask = data.mask.to(device)
            assert x.shape[1]-mask[0].count_nonzero() == model.hole_size
            x = torch.cat((x, mask), -1)
            out = model(x)
            loss = in_loss(out, y, mask)
        else:
            out = model(x)
            loss = in_loss(out, y)

        loss.backward()
        total_loss += loss.item()
        optimizer.step()
        if i == 0:
            pred_mesh = om.TriMesh(
                points=out.detach()[0].numpy(),
                face_vertex_indices=data.face[0].numpy().transpose()
            )
            om.write_mesh(mesh=pred_mesh, filename="train_result.obj")
            label_mesh = om.TriMesh(
                points=y[0].numpy(),
                face_vertex_indices=data.face[0].numpy().transpose()
            )
            om.write_mesh(mesh=label_mesh, filename="train_label.obj")
    return total_loss / len(loader)


def test(model, loader, device, use_mask, in_loss):
    model.eval()
    total_loss = 0
    with torch.no_grad():
        for i, data in enumerate(loader):
            x = data.x.to(device)
            y = data.y.to(device)
            if use_mask:
                mask = data.mask.to(device)
                x = torch.cat((x, mask), -1)
                pred = model(x)
                loss = in_loss(pred, y, mask)
            else:
                pred = model(x)
                loss = in_loss(pred, y)
            total_loss += loss
    return total_loss / len(loader)


def test_mse(model, loader, device, use_mask):
    model.eval()
    total_loss = 0
    with torch.no_grad():
        for i, data in enumerate(loader):
            x = data.x.to(device)
            y = data.y.to(device)
            if use_mask:
                mask = data.mask.to(device)
                x = torch.cat((x, mask), -1)
            pred = model(x)
            f = torch.zeros_like(pred)
            mask = mask.type(torch.bool)
            for e, elem in enumerate(f):
                f[e] = y[e, ~mask[e, :, 0], :]
            loss = F.mse_loss(pred, f)
            total_loss += loss
    return total_loss / len(loader)


def eval_error(model, test_loader, device, meshdata, out_dir, use_mask):
    model.eval()

    errors = []
    mean = meshdata.mean
    std = meshdata.std
    out_data = {}
    with torch.no_grad():
        for i, data in enumerate(test_loader):
            x = data.x.to(device)
            y = data.y.to(device)
            if use_mask:
                mask = data.mask.to(device)
                x = torch.cat((x, mask), -1)
            pred = model(x)
            c = y.clone()
            mask = mask.type(torch.bool)
            for e, elem in enumerate(y):
                c[e, ~mask[e, :, 0], :] = pred[e]
            num_graphs = data.num_graphs
            reshaped_pred = (c.view(num_graphs, -1, 3).cpu() * std) + mean
            reshaped_y = (y.view(num_graphs, -1, 3).cpu() * std) + mean

            reshaped_pred *= 1000
            reshaped_y *= 1000

            tmp_error = torch.sqrt(
                torch.sum((reshaped_pred - reshaped_y)**2,
                          dim=2))  # [num_graphs, num_nodes]
            errors.append(tmp_error)
            if i == 0:
                pred_mesh = om.TriMesh(
                    points=c[0].numpy(),
                    face_vertex_indices=data.face[0].numpy().transpose()
                )
                out_data["output"] = pred_mesh
                #om.write_mesh(mesh=pred_mesh, filename=os.path.join(out_dir,"test_result.obj"))
                label_mesh = om.TriMesh(
                    points=y[0].numpy(),
                    face_vertex_indices=data.face[0].numpy().transpose()
                )
                out_data["label"] = label_mesh
                #om.write_mesh(mesh=label_mesh, filename=os.path.join(out_dir,"test_label.obj"))
                input_mesh = om.TriMesh(
                    points=x[0][:, :3].numpy(),
                    face_vertex_indices=data.face[0].numpy().transpose()
                )
                out_data["input"] = input_mesh
                #om.write_mesh(mesh=input_mesh, filename=os.path.join(out_dir,"test_input.obj"))
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
    return out_data
