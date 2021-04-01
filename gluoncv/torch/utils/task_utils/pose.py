"""Pose utils"""


def train_pose(base_iter,
               model,
               dataloader,
               epoch,
               criterion,
               optimizer,
               cfg,
               writer=None):
    "training pipeline for pose"
    batch_time = AverageMeter()
    data_time = AverageMeter()
    losses = AverageMeter()
    top1 = AverageMeter()
    top5 = AverageMeter()

    model.train()
    end = time.time()
    for step, data in enumerate(dataloader):
        base_iter = base_iter + 1

        train_batch = data[0].cuda()
        train_label = data[1].cuda()
        data_time.update(time.time() - end)

        outputs = model(train_batch)
        loss = criterion(outputs, train_label)
        prec1, prec5 = accuracy(outputs.data, train_label, topk=(1, 5))

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        losses.update(loss.item(), train_label.size(0))
        top1.update(prec1.item(), train_label.size(0))
        top5.update(prec5.item(), train_label.size(0))

        batch_time.update(time.time() - end)
        end = time.time()
        if step % cfg.CONFIG.LOG.DISPLAY_FREQ == 0 and cfg.DDP_CONFIG.GPU_WORLD_RANK == 0:
            print('-------------------------------------------------------')
            for param in optimizer.param_groups:
                lr = param['lr']
            print('lr: ', lr)
            print_string = 'Epoch: [{0}][{1}/{2}]'.format(
                epoch, step + 1, len(dataloader))
            print(print_string)
            print_string = 'data_time: {data_time:.3f}, batch time: {batch_time:.3f}'.format(
                data_time=data_time.val, batch_time=batch_time.val)
            print(print_string)
            print_string = 'loss: {loss:.5f}'.format(loss=losses.avg)
            print(print_string)
            print_string = 'Top-1 accuracy: {top1_acc:.2f}%, Top-5 accuracy: {top5_acc:.2f}%'.format(
                top1_acc=top1.avg, top5_acc=top5.avg)
            print(print_string)
            iteration = base_iter
            writer.add_scalar('train_loss_iteration', losses.avg, iteration)
            writer.add_scalar('train_top1_acc_iteration', top1.avg, iteration)
            writer.add_scalar('train_top5_acc_iteration', top5.avg, iteration)
            writer.add_scalar('train_batch_size_iteration',
                              train_label.size(0), iteration)
            writer.add_scalar('learning_rate', lr, iteration)
    return base_iter

def validate_pose():
    pass
